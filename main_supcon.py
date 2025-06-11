from __future__ import print_function
import os
import sys
import argparse
import time
import math
import torch
import torch.backends.cudnn as cudnn
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import tensorboard_logger as tb_logger
from util import TextAugment, AverageMeter, adjust_learning_rate, warmup_learning_rate, set_optimizer, save_model
from networks.xlmr_supcon import SupConXLMRLarge
from losses import SupConLoss

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--dataset', type=str, default='path', help='dataset name or path')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    opt = parser.parse_args()
    opt.data_folder = './datasets/' if opt.data_folder is None else opt.data_folder
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = [int(it) for it in iterations]
    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.format(
        opt.method, opt.dataset, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    if opt.batch_size > 64:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 1e-6
        opt.warm_epochs = 5
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

class TextDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        if self.transform:
            text1, text2 = self.transform(text)
        else:
            text1, text2 = text, text
        encoding1 = self.tokenizer(text1, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        encoding2 = self.tokenizer(text2, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {
            'input_ids': [encoding1['input_ids'].squeeze(), encoding2['input_ids'].squeeze()],
            'attention_mask': [encoding1['attention_mask'].squeeze(), encoding2['attention_mask'].squeeze()],
            'labels': torch.tensor(label, dtype=torch.long)
        }

def set_loader(opt):
    train_dataset = load_dataset(opt.dataset, split='train') if opt.dataset != 'path' else load_dataset('csv', data_files=opt.data_folder)['train']
    train_transform = TextAugment()
    train_dataset = TextDataset(train_dataset, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    return train_loader

def set_model(opt):
    model = SupConXLMRLarge()
    criterion = SupConLoss(temperature=opt.temp)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_ids = torch.stack(batch['input_ids'], dim=1).cuda(non_blocking=True)  # [bsz, 2, seq_len]
        attention_mask = torch.stack(batch['attention_mask'], dim=1).cuda(non_blocking=True)
        labels = batch['labels'].cuda(non_blocking=True)
        bsz = labels.shape[0]
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        features = []
        for i in range(2):  # Process two views
            feat = model(input_ids[:, i], attention_mask[:, i])
            features.append(feat)
        features = torch.stack(features, dim=1)  # [bsz, 2, feat_dim]
        loss = criterion(features, labels) if opt.method == 'SupCon' else criterion(features)
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
    return losses.avg

def main():
    opt = parse_option()
    train_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()
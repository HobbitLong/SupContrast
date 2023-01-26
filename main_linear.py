from __future__ import print_function

import sys
import argparse
import time
import math
import os
from PIL import Image
import numpy as np
import random
import wandb
import datetime

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=10, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=14, help="number of training epochs"
    )

    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="8,11,13",
        help="where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.2, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    # model dataset
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "path"],
        help="dataset",
    )

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument(
        "--warm", action="store_true", help="warm-up for large batch training"
    )

    parser.add_argument(
        "--ckpt", type=str, default="", help="path to pre-trained model"
    )
    parser.add_argument(
        "--n_cls", type=int, default=0, help="number of classes in the dataset"
    )
    parser.add_argument(
        "--data_folder", type=str, default="./datasets/", help="path to custom dataset"
    )
    parser.add_argument(
        "--val_folder",
        type=str,
        default="./datasets/",
        help="path to custom val dataset",
    )

    opt = parser.parse_args()

    # set the path according to the environment

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = "{}_{}_lr_{}_decay_{}_bsz_{}".format(
        opt.dataset, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size
    )

    if opt.cosine:
        opt.model_name = "{}_cosine".format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = "{}_warm".format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = (
                eta_min
                + (opt.learning_rate - eta_min)
                * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs))
                / 2
            )
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == "cifar10":
        opt.n_cls = 10
    elif opt.dataset == "cifar100":
        opt.n_cls = 100
    elif opt.dataset == "path":
        assert opt.n_cls > 0
    else:
        raise ValueError("dataset not supported: {}".format(opt.dataset))

    opt.model_path = "./save/Linear/{}_models".format(opt.dataset)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location="cpu")
    state_dict = ckpt["model"]

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Take a random image from the batch and save it with its predicted label
            index = int(random.randint(0, len(labels)))
            sample = images[index]
            print("random index", index)
            predicted_label = torch.argmax(output[index])
            save_image(sample, f"data/output/sample_{predicted_label}_{end}.png")

            if idx % opt.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        idx,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

    print(" * Acc@1 {top1.avg:.3f}".format(top1=top1))
    return losses.avg, top1.avg


def save_image(sample, filename):
    """Saves a sample image to disk"""
    sample = sample.cpu().numpy()
    sample = np.transpose(sample, (1, 2, 0))
    sample = (sample - sample.min()) / (sample.max() - sample.min())
    sample = sample * 255
    sample = sample.astype(np.uint8)
    im = Image.fromarray(sample)
    im.save(filename)


def main():
    best_acc = 0
    opt = parse_option()

    # Log
    wandb.login()
    wandb_run = wandb.init(project="supcon", config=vars(opt))
    wandb_run.name = "supcon_linear_" + datetime.datetime.now().strftime(
        "%Y-%m-%d:%Hh%Mm"
    )

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(
            train_loader, model, classifier, criterion, optimizer, epoch, opt
        )
        time2 = time.time()
        print(
            "Train epoch {}, total time {:.2f}, accuracy:{:.2f}".format(
                epoch, time2 - time1, acc
            )
        )
        wandb.log({"train_loss": loss, "train_acc": acc})

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)

        wandb.log({"val_loss": loss, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            save_file = os.path.join(
                opt.save_folder, "ckpt_best_acc_{acc}.pth".format(acc=best_acc)
            )
            save_model(classifier, optimizer, opt, epoch, save_file)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            save_model(classifier, optimizer, opt, epoch, save_file)

    print("best accuracy: {:.2f}".format(best_acc))

    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(classifier, optimizer, opt, opt.epochs, save_file)
    art = wandb.Artifact("supcon_linear", type="model")
    art.add_file(save_file)
    wandb.log_artifact(art)


if __name__ == "__main__":
    main()

import os
import torch
import json
from torch.utils.data import Dataset
from PIL import Image


class BaseQuestionLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root))

    def get_dirname_and_info(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_name = sorted(os.listdir(self.root))[idx]
        json_file_path = os.path.join(self.root, dir_name, f'{dir_name}.json')
        with open(json_file_path) as fh:
            info = json.load(fh)

        return dir_name, info


class Question1Dataset(BaseQuestionLoader):

    def __getitem__(self, idx):

        dir_name, info = self.get_dirname_and_info(idx)

        question_image = info['Questions'][0]['images']
        positive_examples, = [answer['images'] for answer in info['Answers']
                              if answer['group_id'] == info["correct_answer_group_ID"][0]]
        negative_examples, = [answer['images'] for answer in info['Answers']
                              if answer['group_id'] != info["correct_answer_group_ID"][0]]
        samples1, samples2 = [], []
        for im in question_image + positive_examples + negative_examples:
            image = Image.open(os.path.join(
                self.root, dir_name, im['image_url']
            )).convert('RGB')
            sample1, sample2 = self.transform(image)
            samples1.append(sample1)
            samples2.append(sample2)
        samples = [torch.squeeze(torch.stack(samples1), dim=0),
                   torch.squeeze(torch.stack(samples2), dim=0)]

        return samples, torch.tensor([1, 1, 1, 1, 2, 3, 4], dtype=int)


class Question2Dataset(BaseQuestionLoader):

    def __getitem__(self, idx):

        dir_name, info = self.get_dirname_and_info(idx)

        question_image = info['Questions'][0]['images']
        answer_images = info['Answers'][0]['images']
        samples1, samples2 = [], []
        for im in question_image + answer_images:
            image = Image.open(os.path.join(
                self.root, dir_name, im['image_url']
            )).convert('RGB')
            sample1, sample2 = self.transform(image)
            samples1.append(sample1)
            samples2.append(sample2)
        samples = [torch.squeeze(torch.stack(samples1), dim=0),
                   torch.squeeze(torch.stack(samples2), dim=0)]

        return samples, torch.tensor([1, 1, 1, 1 if info["is_correct"] else 2], dtype=int)


class Question3Dataset(BaseQuestionLoader):

    def __getitem__(self, idx):
        dir_name, info = self.get_dirname_and_info(idx)
        positive_samples, negative_examples = [], []

        for question in info['Questions']:
            if question['group_id'] == info["correct_question_group_ID"][0]:
                positive_samples = positive_samples + question['images']
            else:
                negative_examples = negative_examples + question['images']

        for answer in info['Answers']:
            if answer['group_id'] == info["correct_answer_group_ID"][0]:
                positive_samples = positive_samples + answer['images']
            else:
                negative_examples = negative_examples + answer['images']

        samples1, samples2 = [], []
        for im in positive_samples + negative_examples:
            image = Image.open(os.path.join(
                self.root, dir_name, im['image_url']
            )).convert('RGB')
            sample1, sample2 = self.transform(image)
            samples1.append(sample1)
            samples2.append(sample2)
        samples = [torch.squeeze(torch.stack(samples1), dim=0),
                   torch.squeeze(torch.stack(samples2), dim=0)]

        return samples, torch.tensor([1, 1, 1, 1, 1, 1,
                                      2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)


class Question4Dataset(BaseQuestionLoader):

    def __getitem__(self, idx):
        dir_name, info = self.get_dirname_and_info(idx)
        positive_samples, negative_examples = [], []

        for question in info['Questions']:
            positive_samples = positive_samples + question['images']

        for answer in info['Answers']:
            if answer['group_id'] in info["correct_answer_group_ID"]:
                positive_samples = positive_samples + answer['images']
            else:
                negative_examples = negative_examples + answer['images']

        samples1, samples2 = [], []
        for im in positive_samples + negative_examples:
            image = Image.open(os.path.join(
                self.root, dir_name, im['image_url']
            )).convert('RGB')
            sample1, sample2 = self.transform(image)
            samples1.append(sample1)
            samples2.append(sample2)
        samples = [torch.squeeze(torch.stack(samples1), dim=0),
                   torch.squeeze(torch.stack(samples2), dim=0)]

        return samples, torch.tensor([1, 1, 1, 1, 1,
                                      2, 3, 4], dtype=int)

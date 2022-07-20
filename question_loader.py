from __future__ import print_function, division
import os
import torch
import json
from torch.utils.data import Dataset
from PIL import Image


class Question1Dataset(Dataset):
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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_name = sorted(os.listdir(self.root))[idx]
        json_file_path = os.path.join(self.root, dir_name, f'{dir_name}.json')
        with open(json_file_path) as fh:
            info = json.load(fh)
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

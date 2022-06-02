import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()

        # constant
        self.IMG_PER_ID = 4

        # dataset
        self.dataset = []

        self.id_paths = [os.path.join(root, path) for path in os.listdir(root)]
        self.id_num = len(self.id_paths)

        for id, id_dir in enumerate(self.id_paths):
            img_paths = [os.path.join(id_dir, file) for file in os.listdir(id_dir)]
            random.shuffle(img_paths) # inplace operation
            for i in range(0, len(img_paths), self.IMG_PER_ID):
                if i + self.IMG_PER_ID < len(img_paths):
                    img_paths = img_paths[i : i + self.IMG_PER_ID]
                    item = (id, img_paths)
                    self.dataset.append(item)

        # transform
        self.transform = transform
        
    def __getitem__(self, index):
        id, img_paths = self.dataset[index]

        # [4, 3, 256, 256]
        imgs = []
        for path in img_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        labels = torch.tensor(id)

        return imgs, labels
        
    def __len__(self):
        return len(self.dataset)

    def name(self):
        return 'CustomDataset'

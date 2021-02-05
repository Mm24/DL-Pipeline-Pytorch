from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):
        self.data = data.reset_index(drop=True)
        self.mode = mode
        # more transform if model under 0.6
        self.transform_list = [tv.transforms.ToPILImage(), tv.transforms.RandomOrder([tv.transforms.RandomHorizontalFlip(),
                               tv.transforms.RandomRotation(45), tv.transforms.RandomVerticalFlip(), tv.transforms.RandomRotation(25)]),
                               tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std=train_std)]
        #self.transform_list = [tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std=train_std)]
        self._transform = tv.transforms.Compose(self.transform_list)
    pass

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, item):
        if item >= len(self.data): raise IndexError
        image_path = self.data.filename[item]
        image = imread(image_path)
        y = torch.tensor([self.data.crack[item], self.data.inactive[item]])
        sample = gray2rgb(image)
        # transform the data
        if self._transform:
            sample = self._transform(sample)
        return sample, y


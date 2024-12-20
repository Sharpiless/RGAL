from torch.utils.data import Dataset
import glob
import numpy as np
import os
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_and_extract_archive

class TinyImageNet(Dataset):
    def __init__(self, root, split, transform, download=True):

        self.url = "http://cs231n.stanford.edu/tiny-imagenet-200"
        self.root = root
        if download:
            if os.path.exists(f'{self.root}/tiny-imagenet-200/'):
                print(f'{self.root}/tiny-imagenet-200/, File already downloaded')
            else:
                print(f'{self.root}/tiny-imagenet-200/, File isn\'t downloaded')
                download_and_extract_archive(self.url, root, filename="tiny-imagenet-200.zip")

        self.root = os.path.join(self.root, "tiny-imagenet-200")
        self.train = split == "train"
        self.transform = transform
        self.ids_string = np.sort(np.loadtxt(f"{self.root}/wnids.txt", "str"))
        self.ids = {class_string: i for i, class_string in enumerate(self.ids_string)}
        if self.train:
            self.paths = glob.glob(f"{self.root}/train/*/images/*")
            self.targets = [self.ids[path.split("/")[-3]] for path in self.paths]
        else:
            self.paths = glob.glob(f"{self.root}/val/*/images/*")
            self.targets = [self.ids[path.split("/")[-3]] for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = pil_loader(self.paths[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, self.targets[idx]
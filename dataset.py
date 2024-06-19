import torch
from torch.utils.data import Dataset
import os


class CancerDataset(Dataset):

    def __init__(self, image_path):
        self.image_path = image_path
        diagnoses = ['benign', 'malignant', 'normal']
        all_images = [os.listdir(os.path.join(self.image_path, x)) for x in diagnoses]
        self.images = all_images[::3]
        print(all_images)
        print(self.images)

    def __len__(self):
        return

    def __getitem__(self, index):
        return


if __name__ == "__main__":
    cancer_dataset = CancerDataset("dataset")

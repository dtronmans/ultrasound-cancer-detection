import cv2
import torch
from torch.utils.data import Dataset
import os


class CancerDataset(Dataset):

    def __init__(self, image_path, show=False):
        self.image_path = image_path
        self.diagnoses = ['benign', 'malignant', 'normal']
        self.all_images = self._load_images()
        self.show = show

    def _load_images(self):
        """Load and flatten the list of all images from the diagnosis folders."""
        all_images = []
        for diagnosis in self.diagnoses:
            diagnosis_path = os.path.join(self.image_path, diagnosis)
            images = os.listdir(diagnosis_path)
            images = [os.path.join(diagnosis_path, img) for img in images]
            all_images.extend(images)
        return all_images

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_index = self.all_images[(index * 2)]
        label_index = self.all_images[(index * 2 + 1)]
        image = cv2.imread(image_index)
        label = cv2.imread(label_index)

        if self.show:
            self.display_image_label(image, label)

    def display_image_label(self, image, label):
        cv2.imshow("image", image)
        cv2.imshow("label", label)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cancer_dataset = CancerDataset("dataset")
    print(len(cancer_dataset))

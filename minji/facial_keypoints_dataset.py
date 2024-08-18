import os
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
from custom_transforms import Rescale, ToTensor, Normalize
import torchvision.transforms as transforms

class CatKeypointsDataset(Dataset):
    def __init__(self, labels_dir, images_dir, transform=None):
        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.data = self._load_data()
        self.transform = transform

    def _load_data(self):
        data = []
        for label_file in os.listdir(self.labels_dir):
            if label_file.endswith('.json'):
                with open(os.path.join(self.labels_dir, label_file), 'r') as file:
                    labels = json.load(file)
                    img_file = label_file.replace('.json', '.png')
                    data.append({'image': img_file, 'labels': labels['labels'], 'bounding_box': labels['bounding_boxes']})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.data[idx]['image'])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = np.array(self.data[idx]['labels']).astype('float32').reshape(-1, 2)
        bounding_box = np.array(self.data[idx]['bounding_box'])
        sample = {'image': image, 'keypoints': keypoints, 'bounding_box': bounding_box}

        if self.transform:
            sample = self.transform(sample)

        return sample
import torch
import scipy.io as scio
import numpy as np

class MyDataset(torch.utils.data.Dataset): # subclassing
    def __init__(self, data_path, transform=None, train=True):
        self.transform = transform
        self.train = train

        # data follows a dict class, which has keys: train_data, train_labels, etc.
        if self.train:
            self.data = scio.loadmat(data_path)
            self.features = self.data['train_data']
            self.labels = self.data['train_labels']
            self.labels_integer = [np.where(r==1)[0][0] for r in self.labels]
        else:
            self.data = scio.loadmat(data_path.replace('train', 'test'))
            self.features = self.data['test_data']
            self.labels = self.data['test_labels']
            self.labels_integer = [np.where(r == 1)[0][0] for r in self.labels]

    def __getitem__(self, index):
        feature = self.features[index]
        label_integer = self.labels_integer[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label_integer

    def __len__(self):
        return len(self.labels)

    def get_dim(self):
        feature = self.features[0]
        label = self.labels[0]
        return feature.shape[-1], label.shape[-1]



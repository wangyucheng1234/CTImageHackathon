import torch
import pandas as pd
import torch.utils.data as data
import numpy as np

class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
        return label


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

#Following https://keras.io/examples/vision/3D_image_classification/


class MosMed(data.Dataset):
    def __init__(self, datapath, split = 'train'):
        self.datapath = datapath
        self.split = split
        self.target_transform = LabelToLongTensor()
        if self.split == 'train':
            self.csvfile = pd.read_csv(self.datapath + 'train_slices.csv')
        elif self.split == 'val':
            self.csvfile = pd.read_csv(self.datapath + 'test_slices.csv')


    def __getitem__(self, index):
        image_path = self.csvfile.loc[index]['slices_path']
        label_path = self.csvfile.loc[index]['mask_path']
        is_useful = self.csvfile.loc[index]['useful']
        #print(image_path)
        image = np.load(image_path)
        label = np.load(label_path)
        image = normalize(image)
        #print(image_path)
        #volume = sitk.ReadImage(image_path)
        #volume = sitk.Normalize(volume)
        #array = sitk.GetArrayFromImage(volume)
        #array = normalize(array)
        #print(array.shape)
        #array = resize_volume(array)
        #array = np.expand_dims(array, axis = 0)
        label = self.target_transform(label)
        return torch.from_numpy(image).unsqueeze(0), label

    def __len__(self):
        return self.csvfile.shape[0]
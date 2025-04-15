# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .utils import *
import torch.utils.data as data
from torch.utils.data import Dataset,ConcatDataset
from PIL import Image
import glob
from torchvision import transforms as tf
import math
import random
import os
import torch

DATA_DICT= {0: 'galaxy10_merging',
            1: 'galaxy10_disturbed',
            2: 'featured_without_bar_or_spiral',
            3: 'rings'}

def galaxy_feat(test_name,path):
    trainset = torch.load(os.path.join(path,'trainset_512_names.pt'))
    testset = torch.load(os.path.join(path,f'{test_name}_feat_512_names.pt'))
    trainset = CustomDataset(trainset[0],trainset[1],trainset[2])
    testset = CustomDataset(testset[0],testset[1],testset[2])
    
    return [trainset,testset,testset]

class GALDataset(Dataset):
    def __init__(self, root=None, transform=None, phase='train', img_paths = None):
        if root is not None:
            self.img_path = os.path.join(root)
        self.transform = transform
        if img_paths is None:
            self.img_paths, self.labels = self.load_dataset(phase) # self.labels => good : 0, anomaly : 1
        else:
            self.img_paths = img_paths
            self.labels = self.get_labels(phase)
    
    def get_labels(self,phase):
        tot_labels = []
        if phase=='train':
            tot_labels.extend([0]*self.__len__())
        else:
            tot_labels.extend([1]*self.__len__())
        return tot_labels
            
    def load_dataset(self,phase):
        img_tot_paths = []
        tot_labels = []

        img_paths = glob.glob(os.path.join(self.img_path) + "/*.jpg")
        img_paths.sort()
        img_tot_paths.extend(img_paths)
        if phase=='train':
            tot_labels.extend([0]*len(img_paths))
        else:
            tot_labels.extend([1]*len(img_paths))
        
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img,label,img_path.split("/")[-1][:-4]
    
def partial_data_builder(root,class_names, train_tf, val_tf):
    train_img_paths = []
    val_img_paths = []
    
    for name in class_names:
        folder = os.path.join(root,name)
        img_paths = glob.glob(os.path.join(folder) + "/*.jpg")
        train_imgs =  set(random.choices(img_paths, k=math.ceil(len(img_paths)*0.9)))
        val_imgs = set(img_paths) - train_imgs
        train_img_paths.extend(train_imgs)
        val_img_paths.extend(val_imgs)
    train_dataset = GALDataset(transform=train_tf, phase='train', img_paths = train_img_paths)
    val_dataset = GALDataset(transform=val_tf, phase='train', img_paths = val_img_paths)
    
    return train_dataset,val_dataset
    
    
def load_data(data_name,cls,cls_type,norm):
    root = '../dataset'
    if data_name == 'galaxy_feat':
        return galaxy_feat(DATA_DICT[cls], root)
    train_path = os.path.join(root,'mix_normals')
    val_path = os.path.join(root,'val_mix_normals')
    test_path = os.path.join(root,DATA_DICT[cls],'test')
    train_transforms = tf.Compose([tf.CenterCrop(224),
                tf.Resize(28),tf.ToTensor(),
                tf.Normalize(**norm)
                ])
    test_transforms = tf.Compose([tf.CenterCrop(224),
                tf.Resize(28),tf.ToTensor(),
                tf.Normalize(**norm)
                ])
    trainset = GALDataset(root=train_path, transform=train_transforms, phase='train', img_paths = None)
    val_set = GALDataset(root=val_path, transform=test_transforms, phase='train', img_paths = None)
    test_set = GALDataset(root=test_path, transform=test_transforms, phase='test', img_paths = None)
    testset = ConcatDataset([val_set,test_set])
    return [trainset,testset,testset]
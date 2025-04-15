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

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, ConcatDataset
from loader.LoadData import GALDataset
import os

def initialize_model(model_name, use_pretrained=True):
    model_ft = None
    input_size = 0
    model_ft = eval(f'models.{model_name}(pretrained=use_pretrained)')
    input_size = 224
    return model_ft,input_size

def data_transform(input_size):
    return transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def extract_feature(root,train,decals,test_name=None):
    model_ft, input_size = initialize_model('resnet18')
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1]).to('cuda')
    transform = data_transform(input_size)
    if train:
        if decals:
            all_data = []
            for IFol in os.listdir(root):
                images_path = os.path.join(root,IFol)
                dataset = GALDataset(root=images_path, transform=transform, phase='train', img_paths = None)
                all_data.append(dataset)
            dataset = ConcatDataset(all_data)
        else:
            train_path = os.path.join(root,'mix_normals')
            dataset = GALDataset(root=train_path, transform=transform, phase='train', img_paths = None)
    else:
        val_path = os.path.join(root,'val_mix_normals')
        test_path = os.path.join(root,test_name,'test')
        val_set = GALDataset(root=val_path, transform=transform, phase='train', img_paths = None)
        test_set = GALDataset(root=test_path, transform=transform, phase='test', img_paths = None)
        dataset = ConcatDataset([val_set,test_set])

    data_loader = DataLoader(dataset, batch_size=256,shuffle=False)
    data_features = []
    data_targets = []
    data_paths = []

    feature_extractor.eval()
    with torch.no_grad():
        for data,target,path in data_loader:
            data = data.to('cuda')
            feature = feature_extractor(data)
            data_features.append(feature.cpu())
            data_targets.append(target.cpu())
            data_paths.append(path)
            
        data_features = torch.cat(data_features,0).squeeze()
        data_targets = torch.cat(data_targets,0)
        data_paths =  list(sum(data_paths, ()))

    return [data_features,data_targets,data_paths]

if __name__=='__main__':
    root = '../dataset'
    trainset = extract_feature(root,decals=False,train=True)
    torch.save(trainset,'../dataset'+'/trainset_512_names.pt')
    datasets = ["galaxy10_disturbed","galaxy10_merging","featured_without_bar_or_spiral","rings"]
    for name in datasets:
        testset = extract_feature(root,False,False,name)
        torch.save(testset, root+f'/{name}_feat_512_names.pt')

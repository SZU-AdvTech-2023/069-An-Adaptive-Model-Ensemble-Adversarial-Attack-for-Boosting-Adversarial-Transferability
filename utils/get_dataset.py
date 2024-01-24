import os
import torch
import yaml
import pandas as pd
import random
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
# checkpoint yaml file
yaml_path = '../configs/config.yaml'

# 设置随机种子

def get_dataset(args):

    if args.dataset == 'cifar10':
        setattr(args, 'num_classes', 10)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.image_size, antialias=True),
        ])
        test_set = datasets.CIFAR10(root=os.path.join(args.root_path, 'data/cifar10/'), train=False, download=True,
                                    transform=transform_test)
        dataset_length = len(test_set)
        random_indices = random.sample(range(dataset_length), dataset_length // 50)
        subset_test_set = torch.utils.data.Subset(test_set, random_indices)
        test_loader = torch.utils.data.DataLoader(subset_test_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=args.num_worker)
    elif args.dataset == 'cifar100':
        setattr(args, 'num_classes', 100)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.image_size, antialias=True),
        ])
        test_set = datasets.CIFAR100(root=os.path.join(args.root_path, 'data/'), train=False, download=True,
                                     transform=transform_test)
        dataset_length = len(test_set)
        random_indices = random.sample(range(dataset_length), dataset_length // 50)
        subset_test_set = torch.utils.data.Subset(test_set, random_indices)
        test_loader = torch.utils.data.DataLoader(subset_test_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=args.num_worker)
    elif args.dataset == 'imagenet':
        setattr(args, 'num_classes', 1000)
        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
        ])
        test_set = datasets.ImageFolder(root='/root/autodl-tmp/imagenet_val_5k', transform=transform_test)
        dataset_length = len(test_set)
        random_indices = random.sample(range(dataset_length), dataset_length // 50)
        subset_test_set = torch.utils.data.Subset(test_set, random_indices)
        test_loader = torch.utils.data.DataLoader(subset_test_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=args.num_worker)
    elif args.dataset in ['imagenet1000' ,'imagenet1000_adv']:
        with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        path_1k = yaml_data['dataset']['imagenet1000_path']
        path_csv = yaml_data['dataset']['csv_path']
        path_1k = os.path.join(args.root_path, 'utils', path_1k)
        path_csv = os.path.join(args.root_path, 'utils', path_csv)
        setattr(args, 'num_classes', 1000)
        test_set = ImageNet1000(path_1k, path_csv, transforms=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
        ]))
        dataset_length = len(test_set)
        random_indices = random.sample(range(dataset_length), dataset_length // 10)
        subset_test_set = torch.utils.data.Subset(test_set, random_indices)
        test_loader = torch.utils.data.DataLoader(subset_test_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=args.num_worker)
    else:
        raise NotImplemented

    return test_loader

class ImageNet1000(data.Dataset):
    def __init__(self, dir, csv_path, transforms=None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel'] - 1
        # TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        # return data, ImageID, Truelabel
        return data, Truelabel

    def __len__(self):
        return len(self.csv)
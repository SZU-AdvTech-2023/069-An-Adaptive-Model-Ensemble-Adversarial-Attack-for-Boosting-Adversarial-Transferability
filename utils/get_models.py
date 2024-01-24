import os
import timm
import yaml
import torch
from nets import IncV3Ens3, IncV3Ens4, IncResV2Ens
from torchvision import transforms
from utils.AverageMeter import AccuracyMeter
from torch import nn
# checkpoint yaml file

class Normalize(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        (input - mean) / std
        ImageNet normalize:
            'tensorflow': mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            'torch': mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
            
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

def process_model(model, num_classes=100):
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        in_features = model.head.in_features
        if hasattr(model.head, 'fc'):
            if model.head.fc._get_name() == 'Linear':
                model.head.fc = nn.Linear(in_features=in_features, out_features=num_classes)
            else:
                model.head.fc = nn.Conv2d(in_features, num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            model.head = nn.Linear(in_features=in_features, out_features=num_classes)
    else:
        raise ValueError('No fc or classifier or head in model')

def get_models(args, device):
    metrix = {}
    print('🌟\tBuilding models...')
    models = {}
    if args.dataset == 'cifar10':
        save_root_path = r"/root/autodl-tmp/ada_models"
        yaml_path = '../configs/checkpoint_cifar10.yaml'
        with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        for key, value in yaml_data.items():
            # value 以pth 结尾
            if value['ckp_path'].endswith('pth'):
                models[key] = timm.create_model(value['full_name'], pretrained=False)
                process_model(models[key], num_classes=10)
                models[key].load_state_dict(torch.load(os.path.join(save_root_path, yaml_data[key]['ckp_path'])))
                models[key].to(device).eval()
            else:
                models[key] = timm.create_model(value['full_name'],
                                                checkpoint_path=os.path.join(save_root_path,yaml_data[key]['ckp_path']),
                                                num_classes=10).to(device).eval()
            # models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #                                       models[key])
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    elif args.dataset == 'cifar100':
        save_root_path = r"/root/autodl-tmp/ada_models2"
        yaml_path = '../configs/checkpoint_cifar100.yaml'
        with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        for key, value in yaml_data.items():
            if value['ckp_path'].endswith('pth'):
                models[key] = timm.create_model(value['full_name'], pretrained=False)
                process_model(models[key], num_classes=100)
                models[key].load_state_dict(torch.load(os.path.join(save_root_path, yaml_data[key]['ckp_path'])))
                models[key].to(device).eval()
            else:
                models[key] = timm.create_model(value['full_name'],
                                                checkpoint_path=os.path.join(save_root_path,yaml_data[key]['ckp_path']),
                                                num_classes=100).to(device).eval()
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    elif args.dataset == 'imagenet':
        # save_root_path = r"/root/autodl-tmp/ada_models2"
        yaml_path = '../configs/checkpoint_imagenet.yaml'
        with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        for key, value in yaml_data.items():
            model = timm.create_model(value['full_name'], pretrained=True, num_classes=1000).to(device)
            model.eval()
            
            if 'inc' in key or 'vit' in key or 'bit' in key:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), model)
            else:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                  model)
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    elif args.dataset == 'imagenet1000': # 2017 NIPS
        yaml_path = '../configs/checkpoint_imagenet.yaml'
        # yaml_path = '../configs/checkpoint.yaml'
        with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        for key, value in yaml_data.items():
            model = timm.create_model(value['full_name'], pretrained=True, num_classes=1000).to(device)
            # model = timm.create_model(value['full_name'], num_classes=1000).to(device)
            # print(model.default_cfg)
            model.eval()
            if 'inc' in key or 'vit' in key or 'bit' in key:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), model)
            else:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                  model)
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    elif args.dataset == 'imagenet1000_adv': # 对抗训练模型测试
        save_root_path = r"/root/autodl-tmp/ada_models_adv"
        yaml_path = '../configs/checkpoint_imagenet_adv.yaml'
        with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        for key, value in yaml_data.items():
            if value['ckp_path'].endswith('pth'):
                models[key] = timm.create_model(value['full_name'], pretrained=True, num_classes=1000).eval().to(device)
            elif value['ckp_path'].endswith('npy'):
                models[key] = globals()[value['full_name']](weight_file=os.path.join(save_root_path, value['ckp_path']), 
                                                            aux_logits=False)
                models[key] = nn.Sequential(Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), models[key].eval().to(device))
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    
    return models, metrix


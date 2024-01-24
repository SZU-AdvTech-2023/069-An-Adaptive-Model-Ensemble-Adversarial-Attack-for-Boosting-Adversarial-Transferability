import argparse
from email.mime import image
import os

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import timm

from PIL import Image
from sklearn.metrics import average_precision_score
from collections import OrderedDict
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.utils import save_image

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class GCam():
    def __init__(self, ens_models, ens_names, bs=2, dataset='imagenet'):
        target_names = {'resnet18': 'layer4', 
                        'inc_v3': 'Mixed_7c', 
                        'vit_t': 'norm', 
                        'deit_t': 'norm'}
        if dataset in ['imagenet', 'imagenet1000']:
            target_layers = [ getattr(ens_models[i][1] if i < 10 else ens_models[i][1].blocks,
                                    target_names[ens_names[i]]) for i in range(len(ens_models))]
        else:
            target_layers = [ getattr(ens_models[i] if i < 10 else ens_models[i].blocks,
                                    target_names[ens_names[i]]) for i in range(len(ens_models))]
        cams = [ GradCAM(model=ens_models[i], target_layers=[target_layers[i]], use_cuda=True, 
                         reshape_transform = None if i < 2 else reshape_transform)
                for i in range(len(ens_models))]
        for cam in cams:
            cam.batch_size = bs
        self.cams = cams

    def __call__(self, input_tensor):
        cams = self.cams
        cam_masks = []
        for cam in cams:
            cam_mask = cam(input_tensor=input_tensor)
            cam_masks.append(cam_mask)
        # cam_masks = [cams[i](input_tensor=input_tensor) for i in range(len(cams))]
        return cam_masks
    
    
    def cam_on_single(self, img, cam_img):
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        raw_imgs = img * torch.tensor(std).view(3,1,1).cuda() + torch.tensor(mean).view(3,1,1).cuda()

        cam_on_raw = raw_imgs.detach().cpu().numpy()[0, :, :, :]
        cam_on_raw = np.transpose(cam_on_raw, (1, 2, 0))
        cam_on_raw = show_cam_on_image(cam_on_raw, cam_img, use_rgb=True)
        cam_on_raw = np.transpose(cam_on_raw, (2, 0, 1))
        cam_on_raw = torch.from_numpy(cam_on_raw).unsqueeze(0).cuda()
        cam_on_raw = cam_on_raw / torch.max(cam_on_raw)
        return cam_on_raw

    def cam_on_batch(self, imgs, cam_imgs):
        cam_on_raws = [self.cam_on_single(imgs[i], cam_imgs[i]) for i in range(len(imgs))]
        return torch.stack(cam_on_raws, dim=0)

def get_mask(gcam: GCam, imgs):
    cam_masks = gcam(imgs) # [model_num, bs, 224, 224]
    masks = []
    for i in range(len(imgs)):
        cam_masks_now = [torch.from_numpy(cam_masks[j][i]) for j in range(len(cam_masks))]
        cam_masks_now = torch.stack(cam_masks_now, dim=0)
        cam_masks_now = cam_masks_now ** 0.4
        # cam_masks_now[cam_masks_now < cam_masks_now.mean()] = 0
        # mean = cam_masks_now.mean()
        # cam_masks_now[cam_masks_now >= mean * 1.5] = mean * 1.5
        # cam_masks_now = cam_masks_now / mean * 1.5

        # cam_masks_now = cam_masks_now.mean(dim=0)
        cam_masks_now, _ = cam_masks_now.max(dim=0)
        masks.append(cam_masks_now)
    masks = torch.stack(masks, dim=0).unsqueeze(1)
    return masks
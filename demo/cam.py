import argparse
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

# 1. 参数
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_stddev = (0.2023, 0.1994, 0.2010)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 2. 加载数据
trans = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(cifar_mean, cifar_stddev)])
img1 = Image.open('/home/AdaEA/data/image_1k/images/0af0a5dfee6b84ff.png')
img2 = Image.open('/home/AdaEA/data/image_1k/images/0be391239ccba0f2.png')
raw_img1 = trans(img1).unsqueeze(0).to(device)
raw_img2 = trans(img2).unsqueeze(0).to(device)
raw_imgs = torch.cat([raw_img1, raw_img2], dim=0)
print(raw_imgs.shape)

target_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=1000).to(device)

# 3. cam图模型加载
target_layers = [getattr(target_model.blocks, '11')]
cam = GradCAM(model=target_model,
              target_layers=target_layers,
              use_cuda=True,
              reshape_transform=reshape_transform)
cam.batch_size = 1

cam_raw = cam(input_tensor=raw_imgs,
              eigen_smooth=False,
              aug_smooth=False)
print(cam_raw.shape)
cam_raw = cam_raw[1, :]

# 4. 逆归一化
mean=[0.4914, 0.4822, 0.4465]
std=[0.2023, 0.1994, 0.2010]
raw_imgs = raw_imgs * torch.tensor(std).view(3,1,1).cuda() + torch.tensor(mean).view(3,1,1).cuda()

# 5. 生成热力图
cam_on_raw = raw_imgs.detach().cpu().numpy()[0, :, :, :]
cam_on_raw = np.transpose(cam_on_raw, (1, 2, 0))
cam_on_raw = show_cam_on_image(cam_on_raw, cam_raw, use_rgb=True)
cam_on_raw = np.transpose(cam_on_raw, (2, 0, 1))
cam_on_raw = torch.from_numpy(cam_on_raw).unsqueeze(0).cuda()
cam_on_raw = cam_on_raw / torch.max(cam_on_raw)

# 6. 保存cam_raw(已经是numpy格式)
save_image(cam_on_raw, './demo/out/cam_raw.png')
save_image(raw_imgs, './demo/out/raw_imgs.png')

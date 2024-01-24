import os
import numpy as np
import torch

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_project_path():
    """得到项目路径"""
    project_path = os.path.join(
        os.path.dirname(__file__),
        "..",
    )
    return os.path.abspath(project_path)

def norm_forward(x):
    # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
    return (x - mean) / std

def norm_inverse(x):
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
    return x * std + mean

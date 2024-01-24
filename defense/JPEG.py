import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os

trn = transforms.Compose(
    [transforms.ToTensor()]
)


class JPEG_compression(nn.Module):
    # levels=[90,80,70,60,50,40,30]#JPEG compression ratios
    def __init__(self,
                 quality: int = 40,
                 save_path: str = 'JPEG/'):
        super(JPEG_compression, self).__init__()
        self.quality = quality
        self.save_path = save_path

    def forward(self, X_before):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        X_after = torch.zeros_like(X_before)
        for j in range(X_after.size(0)):
            x_np = transforms.ToPILImage()(X_before[j].detach().cpu())
            x_np.save('JPEG/' + 'j.jpg', quality=self.quality)
            X_after[j] = trn(Image.open('JPEG/' + 'j.jpg'))
        return X_after

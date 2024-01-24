# 获取不可见性
import torch

from ignite.engine import Engine
from ignite.metrics import PSNR, SSIM
from utils.DWT import *


def eval_step(engine, batch):
    return batch

class GImper():
    def __init__(self):
        self.num = 0

        self.psnr = 0
        self.default_evaluator_psnr = Engine(eval_step)
        self.psnr_evl = PSNR(data_range=1.2)
        self.psnr_evl.attach(self.default_evaluator_psnr, 'psnr')

        self.ssim = 0
        self.default_evaluator_ssim = Engine(eval_step)
        self.ssim_evl = SSIM(data_range=1.2)
        self.ssim_evl.attach(self.default_evaluator_ssim, 'ssim')

        self.l2 = 0
        self.low_fre = 0

    def update(self, raw_imgs, adv_imgs):
        with torch.no_grad():
            # 计算PSNR
            psnr_state = self.default_evaluator_psnr.run([[raw_imgs, adv_imgs]])
            self.psnr += psnr_state.metrics['psnr'] * raw_imgs.shape[0]

            # 计算SSIM
            ssim_state = self.default_evaluator_ssim.run([[raw_imgs, adv_imgs]])
            self.ssim += ssim_state.metrics['ssim'] * raw_imgs.shape[0]

            # 计算L2 范数
            noise = (adv_imgs - raw_imgs).flatten(start_dim=1)
            self.l2 += torch.sum(torch.norm(noise, p=2, dim=-1))

            # 计算Lwo-fre
            img_ll = DWT_2D_tiny(wavename='haar')(raw_imgs)
            img_ll = IDWT_2D_tiny(wavename='haar')(img_ll)

            adv_ll = DWT_2D_tiny(wavename='haar')(adv_imgs)
            adv_ll = IDWT_2D_tiny(wavename='haar')(adv_ll)

            fre_noise = (adv_ll - img_ll).flatten(start_dim=1)
            self.low_fre += torch.sum(torch.norm(fre_noise, p=2, dim=-1))

            self.num += raw_imgs.shape[0]

    def get_imper(self):
        psnr_value = self.psnr / self.num
        ssim_value = self.ssim / self.num
        l2_value = self.l2 / self.num
        low_fre_value = self.low_fre / self.num
        print(f'psnr: {psnr_value:.2f}  ssim: {ssim_value:.4f}  l2: {l2_value:.2f} low_fre: {low_fre_value:.2f}')
        return self.psnr / self.num, self.ssim / self.num, self.l2 / self.num , self.low_fre / self.num

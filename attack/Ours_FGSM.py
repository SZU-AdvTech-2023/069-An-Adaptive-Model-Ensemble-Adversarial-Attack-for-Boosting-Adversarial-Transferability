"""
AdaEA base on FGSM
"""
import torch
import torch.nn as nn
from attack.Ours_Base import Ours_Base


class Ours_FGSM(Ours_Base):
    def __init__(self, models, eps=8 / 255, max_value=1., min_value=0., threshold=0., beta=10,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), noise_scale=None, N=None):
        super().__init__(models=models, eps=eps, max_value=max_value, min_value=min_value, threshold=threshold,
                         device=device, beta=beta)
        self.attack_step = eps
        self.noise_scale = noise_scale
        self.N = N
    def attack(self, data, label, masks=None):
        B, C, H, W = data.size()
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        data_size = data.size()
        loss_func = nn.CrossEntropyLoss()

        # init pert
        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()
        
        adv_data.requires_grad = True
        outputs_noise_list = []

        # MultSampling
        outputs = [self.models[idx](adv_data) for idx in range(len(self.models))]
        for n in range(self.N):
            if n == 0:
                alpha = self.drf_pro(outputs, outputs_noise_list[n], data_size=(B, C, H, W))
            else:
                alpha += self.drf_pro(outputs, outputs_noise_list[n], data_size=(B, C, H, W))
        alpha /= self.N

        output = torch.stack(outputs, dim=0)
        for n in range(self.N):
            output += torch.stack(outputs_noise_list[n], dim=0)
        output = output * alpha.view(self.num_models, B, 1).detach() ** 2
        output = output.sum(dim=0)

        loss = loss_func(output, label)
        grad = torch.autograd.grad(loss.sum(dim=0), adv_data)[0]

        # Add perturbation
        grad = (grad * masks) if masks is not None else grad

        # add perturbation
        adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad, attack_step=self.eps)
        adv_data.detach_()

        return adv_data


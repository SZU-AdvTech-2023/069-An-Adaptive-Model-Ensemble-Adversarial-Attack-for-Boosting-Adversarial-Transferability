"""
AdaEA base on I-FGSM
"""
import torch
import torch.nn as nn
from attack.Ens_Base import Ens_Base
from utils.tools import norm_forward


class Ens_IFGSM(Ens_Base):
    def __init__(self, models, eps=8/255, alpha=2/255, iters=20, max_value=1., min_value=0., threshold=0.,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), beta=10):
        super().__init__(models=models, eps=eps, alpha=alpha, max_value=max_value, min_value=min_value,
                         threshold=threshold, device=device, beta=beta)
        self.iters = iters

    def attack(self, data, label, idx=-1):
        B, C, H, W = data.size()
        data, label = data.clone().detach().to(self.device), label.clone().detach().to(self.device)
        loss_func = nn.CrossEntropyLoss()

        # init pert
        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()

        for i in range(self.iters):
            adv_data.requires_grad = True

            outputs = [self.models[idx](adv_data) for idx in range(len(self.models))]
            output = torch.stack(outputs, dim=0)
            output = output.sum(dim=0)
            loss = loss_func(output, label)
            grad = torch.autograd.grad(loss.sum(dim=0), adv_data)[0]

            # Add perturbation
            adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad)
            adv_data.detach_()

        return adv_data


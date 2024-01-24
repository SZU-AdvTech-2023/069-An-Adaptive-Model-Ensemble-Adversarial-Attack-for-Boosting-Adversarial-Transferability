"""
AdaEA base on FGSM
"""
import torch
import torch.nn as nn
from attack.SVRE_Base import SVRE_Base


class SVRE_FGSM(SVRE_Base):
    def __init__(self, models, eps=8 / 255, max_value=1., min_value=0., threshold=0., beta=10,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__(models=models, eps=eps, max_value=max_value, min_value=min_value, threshold=threshold,
                         device=device, beta=beta)
        self.attack_step = eps
        
    def attack(self, data, label, idx=-1):
        B, C, H, W = data.size()
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        data_size = data.size()
        loss_func = nn.CrossEntropyLoss()

        # init pert
        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()

        adv_data.requires_grad = True

        outputs = [self.models[idx](adv_data) for idx in range(len(self.models))]
        losses = [loss_func(outputs[idx], label) for idx in range(len(self.models))]
        grads = [torch.autograd.grad(losses[idx], adv_data, retain_graph=True, create_graph=False)[0]
                    for idx in range(len(self.models))]

        output = torch.stack(outputs, dim=0)
        output = output.sum(dim=0)
        loss = loss_func(output, label)
        grad = torch.autograd.grad(loss.sum(dim=0), adv_data)[0]

        x_inner = adv_data.detach()
        grad_inner = torch.zeros_like(x_inner)
        for j in range(self.m_svre):
            x_inner.requires_grad = True
            # 随机挑选一个模型的梯度
            rand_index = torch.randint(0, len(self.models), (1,)).item()
            rand_grad = grads[rand_index]

            # 得到内部迭代x的梯度
            x_inner_output = self.models[rand_index](x_inner)
            x_inner_loss = loss_func(x_inner_output, label)
            x_inner_grad = torch.autograd.grad(x_inner_loss, x_inner, retain_graph=True, create_graph=False)[0]

            noise_inner = x_inner_grad - (rand_grad - grad)
            noise_inner = noise_inner / torch.mean(torch.abs(noise_inner), dim=(1, 2, 3), keepdim=True)

            grad_inner += grad_inner + noise_inner.detach()
            x_inner = self.get_adv_example(ori_data=data, adv_data=x_inner, grad=noise_inner, attack_step=self.eps / 4)

        # add perturbation
        adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad_inner, attack_step=self.eps)
        adv_data.detach_()

        return adv_data


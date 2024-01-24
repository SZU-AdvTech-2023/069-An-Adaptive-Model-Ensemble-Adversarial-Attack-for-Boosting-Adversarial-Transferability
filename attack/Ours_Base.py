"""
Base of the AdaEA
"""
from abc import abstractmethod

import torch
import torch.nn.functional as F


class Ours_Base:
    def __init__(self, models, eps=8/255, alpha=2/255, max_value=1., min_value=0., threshold=0., beta=10,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), no_agm=False, no_drf=False):
        assert isinstance(models, list) and len(models) >= 2, 'Error'
        self.device = device
        self.models = models
        self.num_models = len(self.models)
        for model in models:
            model.eval()

        # attack parameter
        self.eps = eps
        self.threshold = threshold
        self.max_value = max_value
        self.min_value = min_value
        self.beta = beta
        self.alpha = alpha

    def get_adv_example(self, ori_data, adv_data, grad, attack_step=None):
        """
        :param ori_data: original image
        :param adv_data: adversarial image in the last iteration
        :param grad: gradient in this iteration
        :return: adversarial example in this iteration
        """
        if attack_step is None:
            adv_example = adv_data.detach() + grad.sign() * self.alpha
        else:
            adv_example = adv_data.detach() + grad.sign() * attack_step
        delta = torch.clamp(adv_example - ori_data.detach(), -self.eps, self.eps)
        return torch.clamp(ori_data.detach() + delta, max=self.max_value, min=self.min_value)

    
    def drf_pro(self, output, output_noise, data_size):
        sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        # kl散度函数
        reduce_map_result = torch.zeros(size=(self.num_models, data_size[0]), dtype=torch.float, device=self.device)

        for i in range(self.num_models):
            sim = sim_func(F.normalize(output[i].view(data_size[0], -1), dim=1), 
                           F.normalize(output_noise[i].view(data_size[0], -1), dim=1))
            reduce_map_result[i] = 1 + sim

        # 归一化
        return reduce_map_result.softmax(dim=0)
    
    @abstractmethod
    def attack(self,
               data: torch.Tensor,
               label: torch.Tensor,
               idx: int = -1) -> torch.Tensor:
        ...

    def __call__(self, data, label, masks=None):
        return self.attack(data, label, masks=masks)






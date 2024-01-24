import torch
import timm
# 通过
# from robustbench.utils import load_model
# from robustbench.model_zoo import model_dicts as all_models


# keys = all_models.keys()
# value1 = all_models.values()

# for key, value in all_models.items():
#     print(key)
#     for key1, value1 in value.items():
#         print(key1)
#         print(value1)
#     # print(value)
#     print('\n\n')

# 查看timm所有支持的models


# Load a model from the model zoo
# model = load_model(model_name='Liu2023Comprehensive_ConvNeXt-L', dataset='imagenet')
# model = load_model('Engstrom2019Robustness', dataset='cifar10', threat_model='Linf')
# Evaluate the Linf robustness of the model using AutoAttack
# from robustbench.eval import benchmark
# clean_acc, robust_acc = benchmark(model,
#                                   dataset='cifar10',
#                                   threat_model='Linf')


# # 获取所有可用的模型名称
# model_names = timm.list_models()

# # # 打印模型名称列表
# for model_name in model_names:
#     print(model_name)

# 创建模型
# model = timm.create_model('resnetv2_50x1_bit.goog_distilled_in1k', pretrained=True)

# import detectors
import timm

# model = timm.create_model("resnet18_cifar10", pretrained=True)

model = timm.create_model('resnet50',
                  checkpoint_path='/root/autodl-tmp/ada_models/resnet50.pth.tar',
                  num_classes=10).to('cuda').eval()
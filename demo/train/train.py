import torch
import torchvision
import torchvision.transforms as transforms
import timm
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 需要训练的模型：
# 'resnet18', 'inc_v3', 'vit_t', 'deit_t'
# inception_v3
# vit_tiny_patch16_224
# deit_tiny_patch16_224


#  resnet50 
# wide_resnet101_2 
# bit50_1：resnetv2_50x1_bit.goog_in21k_ft_in1k 
# BiT-101：resnetv2_101x1_bit.goog_in21k_ft_in1k
# ViT-B：vit_base_patch16_224
# DeiT-B：deit_base_patch16_224
# Swin-B：swin_base_patch4_window7_224
# Swin-S：swin_small_patch4_window7_224

def process_model(model, num_classes=10):
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        in_features = model.head.in_features
        if hasattr(model.head, 'fc'):
            if model.head.fc._get_name() == 'Linear':
                model.head.fc = nn.Linear(in_features=in_features, out_features=num_classes)
            else:
                model.head.fc = nn.Conv2d(in_features, num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            model.head = nn.Linear(in_features=in_features, out_features=num_classes)
    else:
        raise ValueError('No fc or classifier or head in model')

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def train_and_save_model(model_name, save_path, lr=1e-3):
    dataset_name = 'cifar100'
    dataset = {'cifar10' : 10,
               'cifar100' : 100}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 定义CIFAR-10的数据转换 + 数据增强
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),  # 调整图像大小以匹配模型输入
         transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
         transforms.RandomRotation(10),  # 随机旋转图像
         transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # 随机裁剪图像
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机改变图像的颜色
         transforms.ToTensor(),])
        #  transforms.Normalize(mean, std)

    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    if dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,  # 增加批量大小
                                              shuffle=True, num_workers=32)

    # 修改最后的全连接层以匹配CIFAR-10的类别数 (10) 
    model = timm.create_model(model_name, pretrained=False)
    process_model(model, num_classes=dataset[dataset_name])
    model.load_state_dict(torch.load(f'/root/autodl-tmp/ada_models2/{model_name}_cifar100_aug2.pth'))  # 加载训练好的模型权重
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) # 5e-5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # 训练模型
    for epoch in range(10):  # 增加迭代次数
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, data in progress_bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            progress_bar.set_description(f'Epoch {epoch+1} Loss: {running_loss/(i+1):.4f} Accuracy: {100*correct/total:.2f}%')

        scheduler.step()
        # 保存模型
        torch.save(model.state_dict(), save_path)

    print('Finished Training')



# 使用示例
train_and_save_model('resnet18', 'resnet18_cifar100_aug3.pth', lr=1e-4)
train_and_save_model('inception_v3', 'inception_v3_cifar100_aug3.pth', lr=1e-4)
train_and_save_model('vit_tiny_patch16_224', 'vit_tiny_patch16_224_cifar100_aug3.pth', lr=1e-5)
train_and_save_model('deit_tiny_patch16_224', 'deit_tiny_patch16_224_cifar100_aug3.pth', lr=1e-5)

# train_and_save_model('resnetv2_50x1_bit.goog_in21k_ft_in1k', 'resnetv2_50x1_bit_cifar100.pth')
# train_and_save_model('swin_base_patch4_window7_224', 'swin_base_patch4_window7_224_cifar100.pth')
# train_and_save_model('swin_small_patch4_window7_224', 'swin_small_patch4_window7_224_cifar100.pth')
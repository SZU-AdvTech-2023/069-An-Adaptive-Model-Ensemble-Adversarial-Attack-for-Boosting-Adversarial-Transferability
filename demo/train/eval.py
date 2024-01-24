import torch
import torchvision
import torchvision.transforms as transforms
import timm
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



def test_model(model, testloader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # 将模型设置为评估模式

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():  # 在评估过程中不计算梯度
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# 转换器
def test(model_name, name_tail, dataset):
    base_path = '/root/autodl-tmp/ada_models2/' 
    # base_path = ''
    # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose(
            [transforms.Resize((224, 224)),  # 调整图像大小以匹配模型输入
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    # 加载测试集
    if dataset == 'cifar10':
        num_classes = 10
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar100':
        num_classes = 100
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    # 加载模型并进行测试
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=False)  # 或者你训练的模型名称
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
    model.load_state_dict(torch.load(base_path + model_name +'_' +name_tail + '.pth'))  # 加载训练好的模型权重
    model.to(device)

    test_loss, test_accuracy = test_model(model, testloader, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

dataset = 'cifar100'
name_tail = 'cifar100_norm'
test('resnet18', name_tail, dataset)
test('inception_v3', name_tail, dataset)
test('vit_tiny_patch16_224', name_tail, dataset)
test('deit_tiny_patch16_224', name_tail, dataset)

# 使用示例
# train_and_save_model('resnet18', 'resnet18_cifar10.pth')
# train_and_save_model('inception_v3', 'inception_v3_cifar10.pth')
# train_and_save_model('vit_tiny_patch16_224', 'vit_tiny_patch16_224_cifar10.pth')
# train_and_save_model('deit_tiny_patch16_224', 'deit_tiny_patch16_224_cifar10.pth')

# train_and_save_model('resnetv2_50x1_bit.goog_in21k_ft_in1k', 'resnetv2_50x1_bit_cifar10.pth')
# train_and_save_model('swin_small_patch4_window7_224', 'swin_small_patch4_window7_224_cifar10.pth')

# 
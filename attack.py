from requests import get
import torch
from tqdm import tqdm
from utils.get_attack import get_attack
from utils.get_dataset import get_dataset
from utils.get_models import get_models
from utils.get_mask import GCam, get_mask
from utils.get_imper import GImper
from utils.tools import *
import argparse
import random
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='benchmark of cifar10')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='../checkpoint/',
                        help='the direction to save the dataset')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=('cifar10', 'imagenet1000'))
    parser.add_argument('--batch_size', type=int, default=16,
                        help='the batch size when training')
    parser.add_argument('--image_size', type=int, default=224, # 224
                        help='image size of the dataloader')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether use gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu_id')
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--attack_method', type=str, default='Str_IFGSM',
                        choices=('AdaEA_FGSM', 'AdaEA_IFGSM', 'AdaEA_DIFGSM', 'AdaEA_MIFGSM', 'AdaEA_TIFGSM'))
                        # choices=('SVRE_FGSM', 'SVRE_IFGSM', 'SVRE_DIFGSM', 'SVRE_MIFGSM', 'SVRE_TIFGSM'))
                        # choices=('Ens_FGSM', 'Ens_IFGSM', 'Ens_DIFGSM', 'Ens_MIFGSM', 'Ens_TIFGSM'))
                        # choices=('Str_FGSM', 'Str_IFGSM', 'Str_DIFGSM', 'Str_MIFGSM', 'Str_TIFGSM'))
                        # choices=('Ours_FGSM', 'Ours_IFGSM', 'Ours_DIFGSM', 'Ours_MIFGSM', 'Ours_TIFGSM'))
    parser.add_argument('--fusion_method', type=str, default='add')
    parser.add_argument('--no_norm', action='store_true',
                        help='do not use normalization')
    parser.add_argument('--use_adv_model', action='store_true')
    parser.add_argument('--use_cam_mask', action='store_true')

    # attack parameters
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--alpha', type=float, default=2/255)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='default momentum value')
    parser.add_argument('--resize_rate', type=float, default=0.9,
                        help='resize rate')
    parser.add_argument('--diversity_prob', type=float, default=0.5,
                        help='diversity_prob')
    parser.add_argument('--max_value', type=float, default=1.0)
    parser.add_argument('--min_value', type=float, default=0)

    # AdaEA
    parser.add_argument('--threshold', type=float, default= - 0.3)
    parser.add_argument('--beta', type=float, default=10)
    # Ours
    parser.add_argument('--noise_scale', type=float, default= 0.25) #
    parser.add_argument('--N', type=float, default= 3) #
    parser.add_argument('--gamma', type=float, default= 0.7) #

    args = parser.parse_args()
    args.use_cam_mask = False
    args.save_img = False
    return args

def save_img(imgs, idx, pre_name, args):
    if not args.save_img:
        return 
    path = f'/home/AdaEA/output/{args.attack_method}/{pre_name}'
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i in range(imgs.size(0)):
        num = idx * int(args.batch_size) + i
        img = imgs[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(f'{path}/{num}.png', img)

def main(args):
    # 随机种子
    seed = 42  # 你可以选择任何整数作为种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    device = torch.device(f'cuda:{args.gpu_id}')
    # dataset
    dataloader = get_dataset(args)
    # models
    models, metrix = get_models(args, device=device)
    ens_model = ['resnet18', 'inc_v3', 'vit_t', 'deit_t']
    print(f'ens model: {ens_model}')
    # init cam
    if args.use_cam_mask:
        gcam = GCam(ens_models=[models[i] for i in ens_model], ens_names=ens_model, bs=2, dataset=args.dataset)
    # init imper
    gimper = GImper()

    for idx, (data, label) in enumerate(tqdm(dataloader)):
        n = label.size(0)
        data, label = data.to(device), label.to(device)
        save_img(data, idx, 'clean', args)
        save_img(data + torch.rand_like(data) * args.gamma - args.gamma/2, idx, 'noise1', args)
        save_img(data + torch.rand_like(data) * args.gamma - args.gamma/2, idx, 'noise2', args)
        attack_method = get_attack(args, ens_models=[models[i] for i in ens_model], device=device, models=models)

        if args.use_cam_mask:
            masks1 = get_mask(gcam, data).to(device) # [bs, 1, 224, 224]
            masks2 = get_mask(gcam, data + torch.rand_like(data) * 0.2 - 0.1).to(device) # [bs, 1, 224, 224]
            masks3 = get_mask(gcam, data + torch.rand_like(data) * 0.2 - 0.1).to(device) # [bs, 1, 224, 224]
            masks, _ = torch.stack([masks1 , masks2 , masks3]).max(dim=0)
            masks = torch.where(masks > args.gamma, 1, 0).float()
            # 将masks保存为图片
            adv_exp = attack_method(data, label, masks=masks)
            save_img(masks, idx, 'imper_mask', args)
            save_img(adv_exp, idx, 'imper_adv', args)
            save_img(adv_exp - data, idx, 'imper_noise', args)
        else:
            adv_exp = attack_method(data, label)

            save_img(adv_exp, idx, 'adv', args)
            save_img(adv_exp - data, idx, 'noise', args)

        gimper.update(adv_exp, data)

        for model_name, model in models.items():
            with torch.no_grad():
                r_clean = model(data)
                r_adv = model(adv_exp)
            # clean
            names = ['inc_v3_ens3', 'inc_v3_ens4', 'inc_res_v3_ens']
            pred_clean = r_clean.max(1)[1] - (1 if model_name in names  else 0 ) # 这三个模型标签会移位
            correct_clean = (pred_clean == label).sum().item()
            # adv
            pred_adv = r_adv.max(1)[1] - (1 if model_name in names  else 0 )
            correct_adv = (pred_adv == label).sum().item()
            metrix[model_name].update(correct_clean, correct_adv, n)

    # show result
    avg_clean, avg_adv, avg_asr = 0, 0, 0
    print('-' * 73)
    print('|\tModel name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    for model_name, _ in models.items():
        print(f"|\t{model_name.ljust(10, ' ')}\t"
              f"|\t{str(round(metrix[model_name].clean_acc * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round(metrix[model_name].adv_acc * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round(metrix[model_name].attack_rate * 100, 2)).ljust(8, ' ')}\t|")
        avg_clean += metrix[model_name].clean_acc
        avg_adv += metrix[model_name].adv_acc
        avg_asr += metrix[model_name].attack_rate
    print('-' * 73)
    print(f"|\t{'Average'.ljust(10, ' ')}\t"
          f"|\t{str(round(avg_clean / len(models) * 100, 2)).ljust(13, ' ')}\t"
          f"|\t{str(round(avg_adv / len(models) * 100, 2)).ljust(13, ' ')}\t"
          f"|\t{str(round(avg_asr / len(models) * 100, 2)).ljust(8, ' ')}\t|")
    gimper.get_imper()

if __name__ == '__main__':
    args = get_args()
    same_seeds(args.seed)
    root_path = get_project_path()
    setattr(args, 'root_path', root_path)
    main(args)

from attack import Ens_FGSM, Ens_IFGSM, Ens_MIFGSM, Ens_DIFGSM, Ens_TIFGSM
from attack import AdaEA_FGSM, AdaEA_IFGSM, AdaEA_MIFGSM, AdaEA_DIFGSM, AdaEA_TIFGSM
from attack import SVRE_FGSM, SVRE_IFGSM, SVRE_DIFGSM, SVRE_MIFGSM, SVRE_TIFGSM
from attack import Ours_FGSM, Ours_IFGSM, Ours_MIFGSM, Ours_DIFGSM, Ours_TIFGSM
from attack import Str_FGSM, Str_IFGSM, Str_MIFGSM, Str_DIFGSM
# from attack import Str_IFGSM


def get_attack(args, ens_models, device, models=None):
    # Ens
    if args.attack_method == 'Ens_FGSM':
        attack_method = Ens_FGSM.Ens_FGSM(
            ens_models, eps=args.eps, max_value=args.max_value, min_value=args.min_value, threshold=args.threshold,
            beta=args.beta, device=device)
    elif args.attack_method == 'Ens_IFGSM':
        attack_method = Ens_IFGSM.Ens_IFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
    elif args.attack_method == 'Ens_MIFGSM':
        attack_method = Ens_MIFGSM.Ens_MIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum)
    elif args.attack_method == 'Ens_DIFGSM':
        attack_method = Ens_DIFGSM.Ens_DIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum, resize_rate=args.resize_rate, diversity_prob=args.diversity_prob)
    elif args.attack_method == 'Ens_TIFGSM':
        attack_method = Ens_TIFGSM.Ens_TIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
    # AdaEA
    elif args.attack_method == 'AdaEA_FGSM':
        attack_method = AdaEA_FGSM.AdaEA_FGSM(
            ens_models, eps=args.eps, max_value=args.max_value, min_value=args.min_value, threshold=args.threshold,
            beta=args.beta, device=device)
    elif args.attack_method == 'AdaEA_IFGSM':
        attack_method = AdaEA_IFGSM.AdaEA_IFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
    elif args.attack_method == 'AdaEA_MIFGSM':
        attack_method = AdaEA_MIFGSM.AdaEA_MIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum)
    elif args.attack_method == 'AdaEA_DIFGSM':
        attack_method = AdaEA_DIFGSM.AdaEA_DIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum, resize_rate=args.resize_rate, diversity_prob=args.diversity_prob)
    elif args.attack_method == 'AdaEA_TIFGSM':
        attack_method = AdaEA_TIFGSM.AdaEA_TIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
    # SVRE
    elif args.attack_method == 'SVRE_FGSM':
        attack_method = SVRE_FGSM.SVRE_FGSM(
            ens_models, eps=args.eps, max_value=args.max_value, min_value=args.min_value, threshold=args.threshold,
            beta=args.beta, device=device)
    elif args.attack_method == 'SVRE_IFGSM':
        attack_method = SVRE_IFGSM.SVRE_IFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
    elif args.attack_method == 'SVRE_MIFGSM':
        attack_method = SVRE_MIFGSM.SVRE_MIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum)
    elif args.attack_method == 'SVRE_DIFGSM':
        attack_method = SVRE_DIFGSM.SVRE_DIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum, resize_rate=args.resize_rate, diversity_prob=args.diversity_prob)
    elif args.attack_method == 'SVRE_TIFGSM':
        attack_method = SVRE_TIFGSM.SVRE_TIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
    # Ours
    elif args.attack_method == 'Ours_FGSM':
        attack_method = Ours_FGSM.Ours_FGSM(
            ens_models, eps=args.eps, max_value=args.max_value, min_value=args.min_value, threshold=args.threshold,
            beta=args.beta, device=device, noise_scale=args.noise_scale, N=args.N)
    elif args.attack_method == 'Ours_IFGSM':
        attack_method = Ours_IFGSM.Ours_IFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device, 
            noise_scale=args.noise_scale, N=args.N)
    elif args.attack_method == 'Ours_MIFGSM':
        attack_method = Ours_MIFGSM.Ours_MIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum, noise_scale=args.noise_scale, N=args.N)
    elif args.attack_method == 'Ours_DIFGSM':
        attack_method = Ours_DIFGSM.Ours_DIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum, resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, 
            noise_scale=args.noise_scale, N=args.N)
    elif args.attack_method == 'Ours_TIFGSM':
        attack_method = Ours_TIFGSM.Ours_TIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device, 
            noise_scale=args.noise_scale, N=args.N)
    # Str
    elif args.attack_method == 'Str_FGSM':
        attack_method = Str_FGSM.Str_FGSM(
            ens_models, eps=args.eps, max_value=args.max_value, min_value=args.min_value, threshold=args.threshold,
            beta=args.beta, device=device, noise_scale=args.noise_scale, N=args.N)
    elif args.attack_method == 'Str_IFGSM':
        attack_method = Str_IFGSM.Str_IFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device, 
            noise_scale=args.noise_scale, N=args.N)
    elif args.attack_method == 'Str_MIFGSM':
        attack_method = Str_MIFGSM.Str_MIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum, noise_scale=args.noise_scale, N=args.N)
    elif args.attack_method == 'Str_DIFGSM':
        attack_method = Str_DIFGSM.Str_DIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum, resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, 
            noise_scale=args.noise_scale, N=args.N)
    # elif args.attack_method == 'Ours_TIFGSM':
    #     attack_method = Ours_TIFGSM.Ours_TIFGSM(
    #         ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
    #         min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device, 
    #         noise_scale=args.noise_scale, N=args.N)
    else:
        raise NotImplemented

    return attack_method

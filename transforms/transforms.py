import numpy as np
from torch.nn import fucntional as F
from torchvision import transforms as T



def get_transforms(mean, std, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4,0.4,0.4,0.1),
            T.AutoAugment(),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.5)
        ])
    else:
        return T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])


def mixup_cutmix(images, labels, cfg, return_aug_type=False):
    B = images.size(0)
    device = images.device
    rand_index = torch.randperm(B).to(device)
    aug_type = None
    if np.random.rand() < cfg.switch_prob:
        # CutMix
        lam = np.random.beta(cfg.cutmix_alpha, cfg.cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        aug_type = 'cutmix'
    else:
        # MixUp
        lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
        images = images * lam + images[rand_index] * (1 - lam)
        aug_type = 'mixup'

    y1 = F.one_hot(labels, 200).float()
    y2 = y1[rand_index]
    targets = y1 * lam + y2 * (1 - lam)

    if return_aug_type:
        return images, targets, aug_type
    return images, targets
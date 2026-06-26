import math
import os
import random

import torch
import torchvision.datasets as datasets
from PIL import ImageFilter, ImageOps
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


# 改进 JointCrop
class JointCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
    ):
        super().__init__(size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    def get_params(self, img: Tensor, scale: list[float], ratio: list[float]) -> tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random sized crop.
        """
        width, height = F.get_image_size(img)
        area = height * width
        log_ratio = torch.log(torch.tensor(ratio))
        flag1 = False
        flag2 = False

        log_scale = torch.log(torch.tensor(float(scale[1]) / float(scale[0]) - 0.1))

        # JointCrop 0 -> Uniform distribution
        log_scale_sample = torch.empty(1).uniform_(-log_scale, log_scale)

        scale_sample = torch.exp(log_scale_sample).item()

        target_area1 = area * torch.empty(1).uniform_(
            max(scale[0], scale[0] / scale_sample),
            min(scale[1] / scale_sample, scale[1]),
        )  # scale>1 -> the first item, scale<1 -> the second item
        target_area2 = target_area1 * scale_sample

        for _ in range(10):
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w1 = round(math.sqrt(target_area1 * aspect_ratio))
            h1 = round(math.sqrt(target_area1 / aspect_ratio))

            if 0 < w1 <= width and 0 < h1 <= height:
                i1 = torch.randint(0, height - h1 + 1, size=(1,)).item()
                j1 = torch.randint(0, width - w1 + 1, size=(1,)).item()
                flag1 = True
                break

        for _ in range(10):
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w2 = round(math.sqrt(target_area2 * aspect_ratio))
            h2 = round(math.sqrt(target_area2 / aspect_ratio))

            if 0 < w2 <= width and 0 < h2 <= height:
                i2 = torch.randint(0, height - h2 + 1, size=(1,)).item()
                j2 = torch.randint(0, width - w2 + 1, size=(1,)).item()
                flag2 = True
                break

        if flag1 and flag2:
            return i1, j1, h1, w1, i2, j2, h2, w2
        elif flag1:
            w = width
            h = height
            i = (height - h) // 2
            j = (width - w) // 2
            return i1, j1, h1, w1, i, j, h, w
        elif flag2:
            w = width
            h = height
            i = (height - h) // 2
            j = (width - w) // 2
            return i, j, h, w, i2, j2, h2, w2
        else:
            w = width
            h = height
            i = (height - h) // 2
            j = (width - w) // 2
            return i, j, h, w, i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i1, j1, h1, w1, i2, j2, h2, w2 = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i1, j1, h1, w1, self.size, self.interpolation), F.resized_crop(
            img, i2, j2, h2, w2, self.size, self.interpolation
        )


class TwoCropsTransform_JointCrop:
    """Take two random crops of one image."""

    def __init__(self, base_transform1, base_transform2, scale):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.base_transform = JointCrop(size=224, scale=scale)

    def __call__(self, x):
        im1, im2 = self.base_transform(x)
        im1 = self.base_transform1(im1)
        im2 = self.base_transform2(im2)
        return im1, im2


# --------------------------------------------------------


class Solarize:
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733."""

    def __call__(self, x):
        return ImageOps.solarize(x)


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class MultiCropsTransform:
    """The code is modified from
    https://github.com/maple-research-lab/AdCo/blob/b8f749db3e8e075f77ec17f859e1b2793844f5d3/data_processing/MultiCrop_Transform.py.
    """

    def __init__(self, size_crops, nmb_crops, min_scale_crops, max_scale_crops, normalize, init_size=224):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans = []
        # image_k
        weak = transforms.Compose(
            [
                transforms.RandomResizedCrop(init_size, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        trans.append(weak)
        trans_weak = []

        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )

            weak = transforms.Compose(
                [
                    randomresizedcrop,
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            trans_weak.extend([weak] * nmb_crops[i])

        trans.extend(trans_weak)
        self.trans = trans
        print("in total we have %d transforms" % (len(self.trans)))

    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops


class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index


class CIFAR10Instance(datasets.CIFAR10):
    def __init__(self, root="./", train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        image, _target = super().__getitem__(index)
        return image, index


class CIFAR100Instance(datasets.CIFAR100):
    def __init__(self, root="./", train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        image, _target = super().__getitem__(index)
        return image, index


def build_augmentation(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.multi_crop:
        augmentation = MultiCropsTransform(
            args.size_crops, args.nmb_crops, args.min_scale_crops, args.max_scale_crops, normalize
        )
    else:
        if args.aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        augmentation = TwoCropsTransform(transforms.Compose(augmentation))

    # center-crop augmentation
    eval_augmentation = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )
    return augmentation, eval_augmentation


def cifar10(args):
    augmentation, eval_augmentation = build_augmentation(args)
    train_dataset = CIFAR10Instance(
        root="F:\\contrast-11\\HCSC\\data_visdrone2019", transform=augmentation, download=False
    )
    eval_dataset = CIFAR10Instance(
        root="F:\\contrast-11\\HCSC\\data_visdrone2019", transform=eval_augmentation, download=False
    )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size * 5,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler


def cifar100(args):
    augmentation, eval_augmentation = build_augmentation(args)
    train_dataset = CIFAR100Instance(root="./", transform=augmentation, download=True)
    eval_dataset = CIFAR100Instance(root="./", transform=eval_augmentation, download=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size * 5,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler


def imagenet(args):
    # Data loading code
    if args.debug:
        traindir = os.path.join(args.data, "val")
    else:
        traindir = os.path.join(args.data, "train")

    augmentation, eval_augmentation = build_augmentation(args)

    train_dataset = ImageFolderInstance(traindir, augmentation)
    eval_dataset = ImageFolderInstance(traindir, eval_augmentation)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size * 5,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler

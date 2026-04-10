from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import str_to_pil_interp


DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _dataset_name(data_config) -> str:
    return str(getattr(data_config, "name", "")).lower()


def _normalize(name: str):
    mean, std = DATASET_STATS[name]
    return transforms.Normalize(mean=mean, std=std)


def build_train_transform(data_config):
    name = _dataset_name(data_config)
    image_size = int(getattr(data_config, "image_size", 224))

    if name in {"cifar10", "cifar100"}:
        mean, _ = DATASET_STATS[name]
        hflip = float(getattr(data_config, "hflip", 0.5))
        vflip = float(getattr(data_config, "vflip", 0.0))
        auto_aug = getattr(data_config, "auto_augment", None)
        color_jitter = getattr(data_config, "color_jitter", 0.4)
        re_prob = float(getattr(data_config, "re_prob", 0.0))
        re_mode = str(getattr(data_config, "re_mode", "const"))
        re_count = int(getattr(data_config, "re_count", 1))
        interpolation = str(getattr(data_config, "interpolation", "bilinear"))

        # primary
        primary_tfl = [transforms.RandomCrop(image_size, padding=4)]
        if hflip > 0.:
            primary_tfl.append(transforms.RandomHorizontalFlip(p=hflip))
        if vflip > 0.:
            primary_tfl.append(transforms.RandomVerticalFlip(p=vflip))

        # secondary: AA/RA/AugMix or color jitter
        secondary_tfl = []
        disable_color_jitter = False
        if auto_aug:
            disable_color_jitter = True
            aa_params = dict(
                translate_const=int(image_size * 0.45),
                img_mean=tuple(min(255, round(255 * x)) for x in mean),
            )
            if interpolation != "random":
                aa_params["interpolation"] = str_to_pil_interp(interpolation)
            if auto_aug.startswith("rand"):
                secondary_tfl.append(rand_augment_transform(auto_aug, aa_params))
            elif auto_aug.startswith("augmix"):
                aa_params["translate_pct"] = 0.3
                secondary_tfl.append(augment_and_mix_transform(auto_aug, aa_params))
            else:
                secondary_tfl.append(auto_augment_transform(auto_aug, aa_params))

        if color_jitter is not None and not disable_color_jitter:
            if not isinstance(color_jitter, (list, tuple)):
                color_jitter = (float(color_jitter),) * 3
            secondary_tfl.append(transforms.ColorJitter(*color_jitter))

        # final
        final_tfl = [transforms.ToTensor(), _normalize(name)]
        if re_prob > 0.:
            final_tfl.append(RandomErasing(re_prob, mode=re_mode, max_count=re_count, device="cpu"))

        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

    if name in {"imagenet", "imagefolder", "imagenet_hf"}:
        mean, _ = DATASET_STATS["imagenet"]
        hflip = float(getattr(data_config, "hflip", 0.5))
        auto_aug = getattr(data_config, "auto_augment", None)
        color_jitter = getattr(data_config, "color_jitter", 0.4)
        re_prob = float(getattr(data_config, "re_prob", 0.0))
        re_mode = str(getattr(data_config, "re_mode", "const"))
        re_count = int(getattr(data_config, "re_count", 1))
        interpolation = str(getattr(data_config, "interpolation", "bilinear"))

        primary_tfl = [
            transforms.RandomResizedCrop(image_size),
        ]
        if hflip > 0.:
            primary_tfl.append(transforms.RandomHorizontalFlip(p=hflip))

        secondary_tfl = []
        disable_color_jitter = False
        if auto_aug:
            disable_color_jitter = True
            aa_params = dict(
                translate_const=int(image_size * 0.45),
                img_mean=tuple(min(255, round(255 * x)) for x in mean),
            )
            if interpolation != "random":
                aa_params["interpolation"] = str_to_pil_interp(interpolation)
            if auto_aug.startswith("rand"):
                secondary_tfl.append(rand_augment_transform(auto_aug, aa_params))
            elif auto_aug.startswith("augmix"):
                aa_params["translate_pct"] = 0.3
                secondary_tfl.append(augment_and_mix_transform(auto_aug, aa_params))
            else:
                secondary_tfl.append(auto_augment_transform(auto_aug, aa_params))

        if color_jitter is not None and not disable_color_jitter:
            if not isinstance(color_jitter, (list, tuple)):
                color_jitter = (float(color_jitter),) * 3
            secondary_tfl.append(transforms.ColorJitter(*color_jitter))

        final_tfl = [transforms.ToTensor(), _normalize("imagenet")]
        if re_prob > 0.:
            final_tfl.append(RandomErasing(re_prob, mode=re_mode, max_count=re_count, device="cpu"))

        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def build_eval_transform(data_config):
    name = _dataset_name(data_config)
    image_size = int(getattr(data_config, "image_size", 224))

    if name in {"cifar10", "cifar100"}:
        return transforms.Compose([
            transforms.ToTensor(),
            _normalize(name),
        ])

    if name in {"imagenet", "imagefolder", "imagenet_hf"}:
        resize_size = int(getattr(data_config, "resize_size", int(image_size / 0.875)))
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            _normalize("imagenet"),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def build_event_transform(data_config):
    scale = float(getattr(data_config, "event_scale", 1.0))

    def _transform(frames):
        if isinstance(frames, np.ndarray):
            tensor = torch.from_numpy(frames)
        elif torch.is_tensor(frames):
            tensor = frames
        else:
            tensor = torch.tensor(frames)

        tensor = tensor.float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim == 4 and tensor.shape[-1] in {1, 2} and tensor.shape[1] not in {1, 2}:
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor * scale

    return _transform

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

    if name in {"imagenet", "imagefolder"}:
        resize_size = int(getattr(data_config, "resize_size", int(image_size / 0.875)))
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize("imagenet"),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def build_eval_transform(data_config):
    name = _dataset_name(data_config)
    image_size = int(getattr(data_config, "image_size", 224))

    if name in {"cifar10", "cifar100"}:
        return transforms.Compose([
            transforms.ToTensor(),
            _normalize(name),
        ])

    if name in {"imagenet", "imagefolder"}:
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


def build_dvs_lgn_transform(data_config, is_train: bool = False):
    """Polarity collapse + (optional) magnitude normalization for spikingjelly DVS frames.

    Input: [T_dvs, 2, H_native, W_native] event counts (np.ndarray or tensor).
    Output: [T_dvs, 1, H_out, W_out] float32.

    Config knobs:
        image_size       : target H/W after bilinear (default 0 = keep native).
        polarity_mode    : "signed" (ON-OFF), "mean" (avg), or "stacked" (keep 2-ch).
        magnitude_norm   : "per_sample" (default, ÷ per-sample max),
                           "fixed" (÷ fixed_scale), or "none" (no normalization).
        fixed_scale      : divisor when magnitude_norm == "fixed" (default 8.0,
                           a rough per-pixel max for CIFAR10-DVS event counts).
        dvs_aug          : enable training-time augmentation (default false).
        dvs_aug_hflip    : horizontal-flip probability (default 0.5).
        dvs_aug_crop_pad : pad pixels for random crop (default 4). 0 disables crop.
        dvs_aug_t_shift  : max ± frame offset for random temporal roll (default 2).
                           0 disables. Frames rolled past the edge are zeroed,
                           not wrapped.
    """
    import torch.nn.functional as F  # local import; transforms.py uses torchvision elsewhere

    image_size = int(getattr(data_config, "image_size", 0) or 0)
    polarity_mode = str(getattr(data_config, "polarity_mode", "signed")).lower()
    magnitude_norm = str(getattr(data_config, "magnitude_norm", "per_sample")).lower()
    fixed_scale = float(getattr(data_config, "fixed_scale", 8.0))

    dvs_aug = bool(getattr(data_config, "dvs_aug", False)) and is_train
    aug_hflip = float(getattr(data_config, "dvs_aug_hflip", 0.5))
    aug_crop_pad = int(getattr(data_config, "dvs_aug_crop_pad", 4))
    aug_t_shift = int(getattr(data_config, "dvs_aug_t_shift", 2))
    aug_event_drop = float(getattr(data_config, "dvs_aug_event_drop", 0.0))
    aug_t_cutout = float(getattr(data_config, "dvs_aug_t_cutout", 0.0))
    aug_t_cutout_len = int(getattr(data_config, "dvs_aug_t_cutout_len", 2))

    def _transform(frames):
        if isinstance(frames, np.ndarray):
            tensor = torch.from_numpy(frames)
        elif torch.is_tensor(frames):
            tensor = frames
        else:
            tensor = torch.tensor(frames)
        tensor = tensor.float()
        if tensor.ndim != 4 or tensor.shape[1] != 2:
            raise ValueError(
                f"build_dvs_lgn_transform expects [T, 2, H, W], got {tuple(tensor.shape)}"
            )

        if polarity_mode == "signed":
            collapsed = (tensor[:, 0] - tensor[:, 1]).unsqueeze(1)  # [T, 1, H, W]
        elif polarity_mode == "mean":
            collapsed = tensor.mean(dim=1, keepdim=True)             # [T, 1, H, W]
        elif polarity_mode == "stacked":
            collapsed = tensor                                       # [T, 2, H, W]
        else:
            raise ValueError(
                f"polarity_mode must be 'signed'|'mean'|'stacked', got {polarity_mode!r}"
            )

        if image_size > 0 and (collapsed.shape[-1] != image_size or collapsed.shape[-2] != image_size):
            collapsed = F.interpolate(
                collapsed,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Training-time augmentation: hflip, random crop with padding, random
        # temporal shift. Applied to the resized frames so geometry matches the
        # model's expected input size exactly. Disabled at val/test time.
        if dvs_aug:
            T_aug, C_aug, H_aug, W_aug = collapsed.shape
            if aug_hflip > 0.0 and torch.rand(1).item() < aug_hflip:
                collapsed = collapsed.flip(-1)
            if aug_crop_pad > 0:
                padded = F.pad(collapsed, (aug_crop_pad,) * 4)        # pad H, W on all sides
                max_off = 2 * aug_crop_pad
                off_y = int(torch.randint(0, max_off + 1, (1,)).item())
                off_x = int(torch.randint(0, max_off + 1, (1,)).item())
                collapsed = padded[:, :, off_y:off_y + H_aug, off_x:off_x + W_aug]
            if aug_t_shift > 0:
                shift = int(torch.randint(-aug_t_shift, aug_t_shift + 1, (1,)).item())
                if shift != 0:
                    rolled = torch.zeros_like(collapsed)
                    if shift > 0:
                        rolled[shift:] = collapsed[:-shift]
                    else:
                        rolled[:shift] = collapsed[-shift:]
                    collapsed = rolled

        if magnitude_norm == "per_sample":
            denom = collapsed.abs().amax().clamp_min(1e-3)
            collapsed = collapsed / denom
        elif magnitude_norm == "fixed":
            collapsed = collapsed / max(fixed_scale, 1e-6)
        elif magnitude_norm == "none":
            pass
        else:
            raise ValueError(
                f"magnitude_norm must be 'per_sample'|'fixed'|'none', got {magnitude_norm!r}"
            )
        return collapsed.contiguous()

    return _transform

import torch


def accuracy_at_k(logits, targets, topk=(1, 5)):
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")
    if targets.ndim != 1:
        targets = targets.view(-1)

    with torch.no_grad():
        maxk = min(max(topk), logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        metrics = {}
        for k in topk:
            kk = min(k, logits.size(1))
            correct_k = correct[:kk].reshape(-1).float().sum(0)
            metrics[f"top{k}"] = correct_k / targets.size(0)
        return metrics


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())

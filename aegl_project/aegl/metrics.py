"""
Evaluation metrics matching Table 5.1's benchmarking suite:
    Top-1/5 Accuracy, ECE (calibration), Macro F1, MAE.
"""
import torch


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)):
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    results = {}
    for k in ks:
        k_eff = min(k, logits.size(1))
        correct_k = correct[:k_eff].reshape(-1).float().sum(0)
        results[f"top{k}"] = (correct_k / targets.size(0)).item()
    return results


@torch.no_grad()
def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return ece.item()


@torch.no_grad()
def macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


@torch.no_grad()
def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds - targets).abs().mean().item()

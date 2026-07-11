"""
Runs the full "Ablation Matrix Protocols" (Section 5.1) back-to-back on
the synthetic open-world proxy benchmark and prints a comparison table,
mirroring the structure of Table 2.1 / Table 5.1 in the proposal.

Usage:
    python evaluate.py --epochs 5
"""
import argparse
import torch

from aegl.config import AEGLConfig
from aegl.model import AEGLModel
from aegl.data import build_dataloaders
from aegl.losses import elbo_loss
from aegl.metrics import topk_accuracy, expected_calibration_error, macro_f1


ABLATIONS = ["baseline", "static_topology", "no_elbo", "no_causal", "full"]


def run_one(name, args, device):
    cfg = AEGLConfig.ablation(name, num_classes=args.num_classes, image_size=args.image_size)
    model = AEGLModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader, id_loader, ood_loader = build_dataloaders(
        cfg, samples_per_class=args.samples_per_class, batch_size=args.batch_size
    )

    for _ in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x, run_causal=cfg.use_causal)
            loss, _, _ = elbo_loss(out["logits"], y, out["kl"], cfg.kl_beta, cfg.use_elbo)
            if out["causal_divergence"] is not None:
                loss = loss + 0.01 * out["causal_divergence"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if out["retrieved_idx"] is not None:
                model.memory.update(model.encoder(x).detach(), y, out["retrieved_idx"])

    def eval_split(loader):
        model.eval()
        all_logits, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x, run_causal=False)
                all_logits.append(out["logits"])
                all_probs.append(out["probs"])
                all_targets.append(y)
        logits, probs, targets = torch.cat(all_logits), torch.cat(all_probs), torch.cat(all_targets)
        acc = topk_accuracy(logits, targets, ks=(1,))
        ece = expected_calibration_error(probs, targets)
        f1 = macro_f1(logits, targets, args.num_classes)
        return acc["top1"], ece, f1

    id_acc, id_ece, id_f1 = eval_split(id_loader)
    ood_acc, ood_ece, ood_f1 = eval_split(ood_loader)
    return {
        "ablation": name,
        "id_top1": id_acc, "id_ece": id_ece, "id_f1": id_f1,
        "ood_top1": ood_acc, "ood_ece": ood_ece, "ood_f1": ood_f1,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [run_one(name, args, device) for name in ABLATIONS]

    header = f"{'Ablation':18s} {'ID-Top1':>8s} {'ID-ECE':>8s} {'ID-F1':>8s} {'OOD-Top1':>9s} {'OOD-ECE':>8s} {'OOD-F1':>8s}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['ablation']:18s} {r['id_top1']:8.4f} {r['id_ece']:8.4f} {r['id_f1']:8.4f} "
              f"{r['ood_top1']:9.4f} {r['ood_ece']:8.4f} {r['ood_f1']:8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AEGL ablation matrix (Section 5.1)")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)

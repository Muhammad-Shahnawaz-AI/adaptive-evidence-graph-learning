"""
Training entry point for AEGL -- implements the optimization loop of
Algorithm 1 (backward pass + Adam update, line 12) over the chosen
ablation configuration from Section 5.1.

Usage:
    python train.py --ablation full --epochs 5
    python train.py --ablation baseline
    python train.py --ablation static_topology
    python train.py --ablation no_elbo
    python train.py --ablation no_causal
"""
import argparse
import torch

from aegl.config import AEGLConfig
from aegl.model import AEGLModel
from aegl.data import build_dataloaders
from aegl.losses import elbo_loss
from aegl.metrics import topk_accuracy, expected_calibration_error, macro_f1


def evaluate(model, loader, cfg, device, tag=""):
    model.eval()
    all_logits, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, run_causal=False)
            all_logits.append(out["logits"])
            all_probs.append(out["probs"])
            all_targets.append(y)
    logits = torch.cat(all_logits)
    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)

    acc = topk_accuracy(logits, targets, ks=(1, min(5, cfg.num_classes)))
    ece = expected_calibration_error(probs, targets)
    f1 = macro_f1(logits, targets, cfg.num_classes)
    print(f"[{tag}] top1={acc['top1']:.4f} "
          f"top{min(5, cfg.num_classes)}={acc.get(f'top{min(5, cfg.num_classes)}', float('nan')):.4f} "
          f"ECE={ece:.4f} MacroF1={f1:.4f}")
    return acc, ece, f1


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AEGLConfig.ablation(args.ablation, num_classes=args.num_classes, image_size=args.image_size)
    print(f"Running ablation='{args.ablation}' config={cfg}")

    model = AEGLModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader, id_loader, ood_loader = build_dataloaders(
        cfg, samples_per_class=args.samples_per_class, batch_size=args.batch_size
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_ce, running_kl, running_causal, n_batches = 0.0, 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            out = model(x, run_causal=cfg.use_causal)
            loss, ce, kl_term = elbo_loss(out["logits"], y, out["kl"], cfg.kl_beta, cfg.use_elbo)

            causal_penalty = torch.tensor(0.0, device=device)
            if out["causal_divergence"] is not None:
                causal_penalty = out["causal_divergence"].mean()
                loss = loss + 0.01 * causal_penalty  # small auxiliary auditability penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Slowly refresh memory nodes toward retrieved batch features (non-parametric M)
            if out["retrieved_idx"] is not None:
                model.memory.update(model.encoder(x).detach(), y, out["retrieved_idx"])

            running_ce += ce.item()
            running_kl += kl_term.item()
            running_causal += causal_penalty.item()
            n_batches += 1

        print(f"Epoch {epoch}/{args.epochs} "
              f"CE={running_ce/n_batches:.4f} KL={running_kl/n_batches:.4f} "
              f"CausalDiv={running_causal/n_batches:.4f}")

    print("\n-- Evaluation --")
    evaluate(model, id_loader, cfg, device, tag="ID test (closed-world)")
    evaluate(model, ood_loader, cfg, device, tag="OOD test (open-world shift)")

    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AEGL (Adaptive Evidence Graph Learning)")
    parser.add_argument("--ablation", type=str, default="full",
                         choices=["full", "baseline", "static_topology", "no_elbo", "no_causal"])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="aegl_checkpoint.pt")
    args = parser.parse_args()
    train(args)

"""
AEGL Backend API
================
FastAPI service that wraps the AEGL model (aegl_model.py) and dataset engine
(dataset.py) into a small set of REST endpoints the dashboard frontend talks
to: kick off a training run in the background, poll live progress, and pull
final evaluation results (accuracy, ELBO components, calibration error,
epistemic/aleatoric uncertainty, causal attribution, and per-sample routed
subgraphs for visualization).

Run with:  python main.py
Then open: http://localhost:8000
"""

import io
import threading
import time
import uuid

import numpy as np
import torch
import torch.optim as optim
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from torch.utils.data import DataLoader

from aegl_model import (
    AEGLSystem, AEGLLossEngine, UncertaintyEngine, AEGLCausalVerifier,
    expected_calibration_error,
)
from dataset import LongTailDigitsDataset, UploadedCSVDataset

app = FastAPI(title="AEGL API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

JOBS = {}
JOBS_LOCK = threading.Lock()
UPLOADED_DATASETS = {}  # upload_id -> raw csv bytes


class TrainConfig(BaseModel):
    epochs: int = 12
    batch_size: int = 16
    feature_dim: int = 64
    memory_size: int = 256
    top_k: int = 8
    tau: float = 1.0
    beta: float = 1.0
    lr: float = 1e-3
    imbalance_ratio: float = 8.0
    dataset: str = "longtail_digits"   # or "uploaded"
    upload_id: str | None = None


def _new_job():
    return {
        "status": "queued",       # queued -> running -> done -> error
        "progress": 0.0,
        "epoch": 0,
        "epochs": 0,
        "history": [],             # per-epoch metric dicts
        "results": None,
        "error": None,
        "class_counts": None,
        "num_classes": None,
    }


def _build_datasets(cfg: TrainConfig):
    if cfg.dataset == "uploaded" and cfg.upload_id in UPLOADED_DATASETS:
        raw = UPLOADED_DATASETS[cfg.upload_id]
        full = UploadedCSVDataset(raw)
        n_test = max(1, int(len(full) * 0.25))
        idx = np.random.RandomState(0).permutation(len(full))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        class Subset(torch.utils.data.Dataset):
            def __init__(self, base, indices):
                self.base, self.indices = base, indices
                self.num_classes = base.num_classes
                self.class_counts = base.class_counts

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, i):
                return self.base[self.indices[i]]

        train_set = Subset(full, train_idx)
        test_set = Subset(full, test_idx)
    else:
        train_set = LongTailDigitsDataset(split="train", imbalance_ratio=cfg.imbalance_ratio)
        test_set = LongTailDigitsDataset(split="test", imbalance_ratio=cfg.imbalance_ratio)
    return train_set, test_set


def _run_training_job(job_id: str, cfg: TrainConfig):
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_set, test_set = _build_datasets(cfg)
        num_classes = train_set.num_classes
        in_channels = train_set[0][0].shape[0]

        job["num_classes"] = num_classes
        job["class_counts"] = train_set.class_counts
        job["epochs"] = cfg.epochs

        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        model = AEGLSystem(
            in_channels=in_channels, feature_dim=cfg.feature_dim,
            memory_size=cfg.memory_size, num_classes=num_classes,
            tau=cfg.tau, top_k=cfg.top_k,
        ).to(device)
        criterion = AEGLLossEngine(beta=cfg.beta)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        verifier = AEGLCausalVerifier(alpha=0.15)
        unc_engine = UncertaintyEngine(num_samples=8)

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            ep_loss = ep_ce = ep_kl = 0.0
            n_batches = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                logits, adj = model(images)
                loss, ce, kl = criterion(logits, targets, adj)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ep_loss += loss.item(); ep_ce += ce.item(); ep_kl += kl.item()
                n_batches += 1
            n_batches = max(1, n_batches)

            # quick train-accuracy read (no grad) for the live chart
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, targets in train_loader:
                    images, targets = images.to(device), targets.to(device)
                    logits, _ = model(images)
                    correct += (logits.argmax(-1) == targets).sum().item()
                    total += targets.size(0)
            train_acc = 100.0 * correct / max(1, total)

            job["history"].append({
                "epoch": epoch, "loss": ep_loss / n_batches, "ce": ep_ce / n_batches,
                "kl": ep_kl / n_batches, "train_accuracy": train_acc,
            })
            job["epoch"] = epoch
            job["progress"] = epoch / cfg.epochs

        # ---- final evaluation pass ----
        model.eval()
        test_correct, total_samples = 0, 0
        all_probs, all_targets = [], []
        causal_scores = []
        epistemic_list, aleatoric_list = [], []
        sample_gallery = []

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                logits, adj = model(images)
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(-1)

                test_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)
                all_probs.append(probs)
                all_targets.append(targets)

                causal = verifier(model, images, logits, adj)
                causal_scores.extend(causal.cpu().tolist())

            # uncertainty + sample gallery on a capped subset of test data for speed
            gallery_n = min(24, len(test_set))
            gallery_imgs = torch.stack([test_set[i][0] for i in range(gallery_n)]).to(device)
            gallery_labels = torch.stack([test_set[i][1] for i in range(gallery_n)])
            unc = unc_engine.estimate(model, gallery_imgs)

            epistemic_list = unc["epistemic_uncertainty"].cpu().tolist()
            aleatoric_list = unc["aleatoric_uncertainty"].cpu().tolist()
            gallery_preds = unc["mean_probs"].argmax(-1).cpu().tolist()
            gallery_conf = unc["mean_probs"].max(-1).values.cpu().tolist()
            gallery_adj = unc["mean_adjacency"].cpu()

            for i in range(gallery_n):
                topk = torch.topk(gallery_adj[i], k=min(6, gallery_adj.size(-1)))
                pixels = gallery_imgs[i, 0].cpu().flatten().tolist()
                side = gallery_imgs.shape[-1]
                sample_gallery.append({
                    "index": i,
                    "true_label": int(gallery_labels[i].item()),
                    "pred_label": int(gallery_preds[i]),
                    "confidence": float(gallery_conf[i]),
                    "epistemic": float(epistemic_list[i]),
                    "aleatoric": float(aleatoric_list[i]),
                    "pixels": pixels,
                    "side": side,
                    "top_memory_edges": [
                        {"node": int(idx), "weight": float(w)}
                        for w, idx in zip(topk.values.tolist(), topk.indices.tolist())
                    ],
                })

        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        ece, bin_stats = expected_calibration_error(all_probs, all_targets)

        job["results"] = {
            "test_accuracy": 100.0 * test_correct / max(1, total_samples),
            "test_size": total_samples,
            "ece": ece,
            "calibration_bins": bin_stats,
            "mean_causal_attribution": float(np.mean(causal_scores)) if causal_scores else 0.0,
            "mean_epistemic_uncertainty": float(np.mean(epistemic_list)) if epistemic_list else 0.0,
            "mean_aleatoric_uncertainty": float(np.mean(aleatoric_list)) if aleatoric_list else 0.0,
            "sample_gallery": sample_gallery,
            "class_counts": train_set.class_counts,
        }
        job["status"] = "done"
        job["progress"] = 1.0

    except Exception as e:  # noqa
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    upload_id = str(uuid.uuid4())
    UPLOADED_DATASETS[upload_id] = content
    try:
        preview = UploadedCSVDataset(content)
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    return {
        "upload_id": upload_id,
        "num_samples": len(preview),
        "num_classes": preview.num_classes,
        "class_counts": preview.class_counts,
    }


@app.post("/api/train")
def start_training(cfg: TrainConfig):
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = _new_job()
    thread = threading.Thread(target=_run_training_job, args=(job_id, cfg), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "status": job["status"], "progress": job["progress"], "epoch": job["epoch"],
        "epochs": job["epochs"], "history": job["history"], "error": job["error"],
        "class_counts": job["class_counts"], "num_classes": job["num_classes"],
    }


@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"job status is '{job['status']}', not ready")
    return job["results"]


# ---- serve the frontend as static files ----
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

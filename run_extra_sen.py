# http://extrasensory.ucsd.edu/#papers
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mlp import MLP
from sech_kan import SechKAN
from efficient_kan import EfficientKAN
from fast_kan import FastKAN
from bsrbf_kan import BSRBF_KAN
import tabm

from schedulers import get_scheduler
from utils import count_params, set_seed

FEATURES_URL = "http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip"
CV_URL = "http://extrasensory.ucsd.edu/data/cv5Folds.zip"
N_FEATURES, N_LABELS = 225, 51
METADATA = {"timestamp", "uuid", "label_source"}
MAIN_ACTIVITIES = ["LYING_DOWN", "SITTING", "OR_standing", "FIX_walking", "FIX_running", "BICYCLING"]


# Download / files
def find(root, pattern):
    return sorted(p for p in Path(root).rglob(pattern)
                    if "__MACOSX" not in p.parts and not p.name.startswith("._"))


def safe_extract(zip_path, out_dir):
    """Extract zip, refusing members that would escape out_dir."""
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            if not (out_dir / m.filename).resolve().is_relative_to(root):
                raise RuntimeError(f"Unsafe zip member: {m.filename}")
        zf.extractall(out_dir)


def prepare_data(root):
    """Download features + official folds unless already somewhere under root."""
    root.mkdir(parents=True, exist_ok=True)
    for name, url, pattern in [("features", FEATURES_URL, "*.features_labels.csv.gz"),
                               ("cv5Folds", CV_URL, "fold_*_uuids.txt")]:
        if find(root, pattern):
            continue
        zip_path = root / f"{name}.zip"
        if not zip_path.exists():
            print(f"Downloading {url}")
            part = zip_path.with_suffix(".part")  # no corrupt zip if download is interrupted
            urllib.request.urlretrieve(url, part)
            part.replace(zip_path)
        safe_extract(zip_path, root / name)
        zip_path.unlink()
        if not find(root, pattern):
            raise FileNotFoundError(f"No '{pattern}' files under {root} after extracting {url}")


def uuid_of(path):
    return Path(path).name.split(".")[0].lower()  # "<UUID>.features_labels.csv.gz" -> "<uuid>"


# Parsing + cache
def read_user(path):
    """Read one user file with strict schema checks."""
    df = pd.read_csv(path)
    labels = [c for c in df.columns if c.startswith("label:")]
    features = [c for c in df.columns if c not in METADATA and c not in labels]
    if len(features) != N_FEATURES or len(labels) != N_LABELS:
        raise ValueError(f"{path.name}: expected {N_FEATURES} features / {N_LABELS} labels, "
                         f"found {len(features)} / {len(labels)}")
    X = df[features].to_numpy(np.float32)  # raises on non-numeric values
    Y = df[labels].to_numpy(np.float32)
    bad = np.isfinite(Y) & ~np.isin(Y, (0.0, 1.0))
    if bad.any():
        raise ValueError(f"{path.name}: labels must be 0/1/NaN, found {np.unique(Y[bad])[:5]}")
    return X, Y, features, [c[len("label:"):] for c in labels]


def load_all_users(root):
    """Parse every user once; cache keyed on file names, sizes and mtimes."""
    files = find(root, "*.features_labels.csv.gz")
    key = hashlib.sha256("|".join(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}"
                                  for f in files).encode()).hexdigest()[:16]
    cache = root / f"extrasensory_cache_{key}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return d["X"], d["Y"], d["users"], d["features"].tolist(), d["labels"].tolist()

    Xs, Ys, Us, schema = [], [], [], None
    for f in tqdm(files, desc="Parsing users (first run only)"):
        X, Y, features, labels = read_user(f)
        if schema is None:
            schema = (features, labels)
        elif (features, labels) != schema:
            raise ValueError(f"{f.name}: column order differs from other users")
        Xs.append(X); Ys.append(Y); Us.append(np.full(len(X), uuid_of(f)))

    X, Y, users = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Us)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(tmp, X=X, Y=Y, users=users, features=np.array(schema[0]), labels=np.array(schema[1]))
    tmp.replace(cache)
    return X, Y, users, schema[0], schema[1]


# Official split
def read_fold(root, fold, split):
    """Union of fold_{k}_{split}_android_uuids.txt and fold_{k}_{split}_iphone_uuids.txt."""
    files = find(root, f"fold_{fold}_{split}_*.txt")
    if not files:
        raise FileNotFoundError(f"No 'fold_{fold}_{split}_*.txt' under {root}")
    return sorted({line.strip().lower() for f in files for line in open(f) if line.strip()})


def split_users(root, fold, val_fraction):
    train_pool, test = read_fold(root, fold, "train"), read_fold(root, fold, "test")
    if set(train_pool) & set(test):
        raise RuntimeError(f"Fold {fold}: train/test users overlap")
    pool = np.array(train_pool)
    np.random.default_rng(42).shuffle(pool) # seed_split = 42
    n_val = max(1, int(np.ceil(len(pool) * val_fraction)))
    if n_val >= len(pool):
        raise ValueError("Validation split leaves no training users")
    return sorted(pool[n_val:].tolist()), sorted(pool[:n_val].tolist()), test


# Dataset
def load_dataset(args):
    
    root = Path(args.data_root)
    prepare_data(root)
    X, Y, users, feature_names, label_names = load_all_users(root)

    # User split; every user in the fold files must exist in the data
    splits = split_users(root, args.fold, args.val_subject_fraction)
    missing = set(sum(splits, [])) - set(np.unique(users))
    if missing:
        raise FileNotFoundError(f"Users in fold files but missing from data: {sorted(missing)[:5]}")
    
    
    if args.task == "main":
        P = Y[:, [label_names.index(a) for a in MAIN_ACTIVITIES]] == 1
        n_pos = P.sum(1)
        """
            n_pos=0: 70752
            n_pos=1: 306594
            n_pos>1: 0
        """
    
        keep = n_pos == 1
        X, users, y = X[keep], users[keep], P[keep].argmax(1)
        m, class_names = None, MAIN_ACTIVITIES
    else:
        m, y, class_names = np.isfinite(Y), np.nan_to_num(Y), label_names

    tr, va, te = [np.isin(users, s) for s in splits]

    # Preprocessing fitted on TRAIN only (input dim stays 225)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        med, mean, std = (np.nan_to_num(f(X[tr], axis=0)) for f in (np.nanmedian, np.nanmean, np.nanstd))
    std[std < 1e-8] = 1.0

    def prep(a):
        a = (np.where(np.isfinite(a), a, med) - mean) / std
        return torch.from_numpy(np.clip(a, -args.clip, args.clip) if args.clip > 0 else a).float()

    def part(sel):
        if m is None:
            return [prep(X[sel]), torch.from_numpy(y[sel]).long()]
        return [prep(X[sel]), torch.from_numpy(y[sel]).float(), torch.from_numpy(m[sel]).float()]

    data = [part(s) for s in (tr, va, te)]
    print(f"ExtraSensory ({args.task}, fold {args.fold}) | "
          f"users train/val/test: {len(splits[0])}/{len(splits[1])}/{len(splits[2])} | "
          f"examples: {tr.sum():,}/{va.sum():,}/{te.sum():,} | "
          f"features: {len(feature_names)} | classes: {len(class_names)}")
    print(f"Validation users: {splits[1]}")
    return data, class_names, [len(s) for s in splits]


# Model/loss/metrics
def build_model(args, input_dim, num_classes):
    hidden = [int(h) for h in args.hidden_layers.split(",") if h.strip()]
    if not hidden or min(hidden) <= 0:
        raise ValueError("--hidden_layers must be positive integers, e.g. '256' or '256,128'")
    layers = [input_dim] + hidden + [num_classes]
    if args.model == "mlp":
        return MLP(net_layers=layers, base_activation=args.activation, norm_type=args.norm_type)
    if args.model == "sech_kan":
        return SechKAN(net_layers=layers, num_grids=args.num_grids, use_base_update=False, base_activation="silu",
                       norm1_type=args.norm1_type, norm2_type=args.norm2_type, norm_mode=args.norm_mode,
                       use_width=False, net_type="standard")  
    if args.model == 'fast_kan':
        return FastKAN(layers_hidden=layers, num_grids=args.num_grids, use_base_update=True, 
                       use_layernorm=True) 

    if args.model == 'bsrbf_kan':
        return BSRBF_KAN(layers_hidden=layers, grid_size=5, spline_order=3, norm_type=args.norm_type)

    if args.model == "tab_m":
        
        if args.task == "main":
            return tabm.TabM.make(n_num_features=225, cat_cardinalities=[], d_out=6, d_block=200, n_blocks=1, k=8, num_embeddings=None)
        else: # multilabel, 71002
            return tabm.TabM.make(n_num_features=225, cat_cardinalities=[], d_out=51, d_block=106, n_blocks=1, k=8, num_embeddings=None)

    if args.model == "efficient_kan":
        return EfficientKAN(layers_hidden=layers, grid_size=5, spline_order=3)      
        
    raise  ValueError(f"Unknown model: {args.model}")


def class_weights(args, train_t, C):
    """Training-set weights for --balance: CE class weights (main) or BCE pos_weight (multilabel)."""
    if not args.balance:
        return None
    if args.task == "main":
        counts = torch.bincount(train_t[1], minlength=C).float()
        return (counts.sum() / (C * counts.clamp_min(1))).to(device)
    y, m = train_t[1], train_t[2]
    return (((1 - y) * m).sum(0) / (y * m).sum(0).clamp_min(1)).clamp(1, 50).to(device)


def loss_fn(task, out, y, m, weight=None):
    if task == "main":
        return F.cross_entropy(out.float(), y, weight=weight)
    loss = F.binary_cross_entropy_with_logits(out.float(), y, pos_weight=weight, reduction="none")
    return (loss * m).sum() / m.sum().clamp_min(1)  # missing labels do not contribute


def compute_metrics(task, out, y, m, C, threshold):
    """
    Per-label counts over observed entries only.
    - macro F1 uses labels with >= 1 positive
    - balanced accuracy uses labels with >= 1 positive AND >= 1 negative
      (main task: mean per-class recall)
    """
    if task == "main":
        P, T, M = F.one_hot(out.argmax(1), C).float(), F.one_hot(y, C).float(), torch.ones(len(y), C)
    else:
        P, T, M = (torch.sigmoid(out) >= threshold).float(), y, m
    tp, fp = (P * T * M).sum(0), (P * (1 - T) * M).sum(0)
    fn, tn = ((1 - P) * T * M).sum(0), ((1 - P) * (1 - T) * M).sum(0)
    pos, neg = tp + fn, tn + fp
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    if task == "main":  # single-label BA = mean per-class recall
        ba = tp / pos.clamp_min(1)
    else:               # multi-label BA = mean per-label (TPR + TNR) / 2
        ba = 0.5 * (tp / pos.clamp_min(1) + tn / neg.clamp_min(1))
    has_pos, has_both = pos > 0, (pos > 0) & (neg > 0)

    res = {
        "balanced_accuracy": ba[has_both].mean().item() if has_both.any() else float("nan"),
        "macro_f1": f1[has_pos].mean().item() if has_pos.any() else float("nan"),
        "micro_f1": (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp_min(1)).item(),
        "labels_evaluated": int(has_both.sum()),
        "per_label": {"f1": f1.tolist(), "balanced_accuracy": ba.tolist(),
                      "positives": pos.long().tolist(), "evaluated": has_both.tolist()},
    }
    if task == "main":
        res["accuracy"] = (out.argmax(1) == y).float().mean().item()
    return res


# Train/evaluate
def run_epoch(model, loader, args, C, weight=None, optimizer=None, scaler=None, scheduler=None, per_batch=False):
    """One pass over loader: trains if optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    loss_sum, n_sum, outs, ys, ms = 0.0, 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for batch in (tqdm(loader, desc="Training", leave=False) if train else loader):
            x, y, *m = [b.to(device, non_blocking=True) for b in batch]
            m = m[0] if m else None

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                if args.model == "tab_m":
                    loss, out = tabm_loss_and_prediction(out, y, m, args.task, weight, train)
                else:
                    # All other models: [B, C]
                    loss = loss_fn(args.task, out, y, m, weight)

            if train:
                optimizer.zero_grad(set_to_none=True)
                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)  # GradScaler skips steps with inf/nan grads
                scaler.update()
                if per_batch and scaler.get_scale() >= old_scale:  # step LR only if optimizer stepped
                    scheduler.step()

            n = m.sum().item() if m is not None else y.size(0)  # weight by observed labels
            loss_sum += loss.item() * n
            n_sum += n
            outs.append(out.detach().float().cpu())
            ys.append(y.cpu())
            if m is not None:
                ms.append(m.cpu())

    res = compute_metrics(args.task, torch.cat(outs), torch.cat(ys), torch.cat(ms) if ms else None,
                          C, args.threshold)
    res["loss"] = loss_sum / max(n_sum, 1)
    return res


def make_scheduler(optimizer, args, steps):
    cfg = {
        "StepLR": (dict(step_size=max(args.epochs // 3, 1)), False),
        "CosineAnnealingLR": (dict(epochs=args.epochs), False),
        "OneCycleLR": (dict(step_size=steps * args.epochs), True),
        "ExponentialLR": ({}, False),
        "CyclicLR": (dict(step_size=steps * 2), True),
    }
    kwargs, per_batch = cfg[args.scheduler]
    return get_scheduler(optimizer, name=args.scheduler, **kwargs), per_batch


def scalars(r, prefix=""):
    return {f"{prefix}{k}": v for k, v in r.items() if k != "per_label"}


def tabm_loss_and_prediction(out, y, m, task, weight, train):
    k = out.size(1)
    if task == "main":
        probs = torch.softmax(out.float(), -1).mean(1)
        pred_out = torch.log(probs.clamp_min(1e-6))
        if train:
            loss = loss_fn(task, out.flatten(0, 1), y.repeat_interleave(k, 0), None, weight)
        else:
            loss = F.nll_loss(pred_out, y, weight=weight)
    else:
        probs = torch.sigmoid(out.float()).mean(1)
        pred_out = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
        if train:
            m_rep = m.repeat_interleave(k, 0) if m is not None else None
            loss = loss_fn(task, out.flatten(0, 1), y.repeat_interleave(k, 0), m_rep, weight)
        else:
            loss = loss_fn(task, pred_out, y, m, weight)
    return loss, pred_out
    
def train_model(args, data, class_names, n_users):
    C = len(class_names)
    train_t = data[0]
    loaders = [DataLoader(TensorDataset(*t), batch_size=args.batch_size, shuffle=(i == 0),
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.num_workers > 0) for i, t in enumerate(data)]
    train_loader, val_loader, test_loader = loaders

    model = build_model(args, train_t[0].shape[1], C).to(device)
    weight = class_weights(args, train_t, C) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, per_batch = make_scheduler(optimizer, args, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path(args.output_root) / f"extrasensory_{args.task}" / f"fold_{args.fold}" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}__{args.note or 'run'}__seed_{args.seed}"
    #stem = f"{args.model}__{args.note or 'run'}"
    ckpt_path, history_path, summary_path = (out_dir / f"{stem}{s}" for s in (".pt", ".jsonl", ".json"))
    history_path.unlink(missing_ok=True)
    ckpt_path.unlink(missing_ok=True)

    sel = args.select_metric
    best, best_epoch = -np.inf, 0
    start = time.perf_counter()
    print(f"\nModel: {args.model} | select: val {sel}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_loader, args, C, weight, optimizer, scaler, scheduler, per_batch)
        if not per_batch:
            scheduler.step()
        va = run_epoch(model, val_loader, args, C, weight)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt, lr = time.perf_counter() - t0, optimizer.param_groups[0]["lr"]

        if np.isfinite(va[sel]) and va[sel] > best:
            best, best_epoch = va[sel], epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "metric": best,
                        "args": vars(args), "class_names": list(class_names)}, ckpt_path)

        with history_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch, "lr": lr, "seconds": dt, "best_epoch": best_epoch,
                                **scalars(tr, "train_"), **scalars(va, "val_")}) + "\n")

        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr['loss']:.4f} | val loss {va['loss']:.4f} | "
              f"BA {va['balanced_accuracy'] * 100:.2f}% | val macro-F1 {va['macro_f1'] * 100:.2f}% | "
              f"val micro-F1 {va['micro_f1'] * 100:.2f}% | lr {lr:.3g} | {dt:.1f}s")

    # Test with the best validation checkpoint
    if best_epoch == 0:
        raise RuntimeError(f"No valid checkpoint: val {sel} was NaN in every epoch")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    te = run_epoch(model, test_loader, args, C)
    total = time.perf_counter() - start

    pl = te["per_label"]
    summary = {
        "dataset": "ExtraSensory", "args": vars(args), "users_train_val_test": n_users,
        "input_dim": int(train_t[0].shape[1]), "num_classes": C, "parameters": int(count_params(model)),
        "best_epoch": best_epoch, f"best_val_{sel}": best, "total_seconds": total,
        "test": scalars(te),
        "test_per_label": {name: {"f1": pl["f1"][i], "balanced_accuracy": pl["balanced_accuracy"][i],
                                  "positives": pl["positives"][i], "evaluated": pl["evaluated"][i]}
                           for i, name in enumerate(class_names)},
    }
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(summary_path)

    print(f"\n{'=' * 64}\nTask {args.task} | fold {args.fold} | {args.model} | seed {args.seed}\n"
          f"Best epoch {best_epoch} (val {sel} {best * 100:.2f}%)\n"
          + "".join(f"Test {k:<18}: {v * 100:.2f}%\n" for k, v in scalars(te).items()
                    if k in ("balanced_accuracy", "macro_f1", "micro_f1", "accuracy"))
          + f"Labels evaluated  : {te['labels_evaluated']}/{C}\nTime: {total:.1f}s\n{'=' * 64}")
    return summary


# Main
def get_args():
    p = argparse.ArgumentParser(description="ExtraSensory benchmark (225 features / 51 labels)")
    p.add_argument("--model", default="mlp", choices=["mlp", "sech_kan", "efficient_kan", "fast_kan", "bsrbf_kan", "tab_m"])
    p.add_argument("--task", default="multilabel", choices=["multilabel", "main"],
                   help="multilabel: 51 labels, masked BCE | main: 6 main activities, CE")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--output_root", default="./output")
    p.add_argument("--note", default="")

    # Split / preprocessing
    p.add_argument("--fold", type=int, default=0, choices=range(5), help="official fold 0-4")
    p.add_argument("--val_subject_fraction", type=float, default=0.2)
    
    p.add_argument("--clip", type=float, default=5.0, help="clip standardized features (0 = off)")

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden_layers", default="256")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", default="OneCycleLR",
                   choices=["StepLR", "CosineAnnealingLR", "OneCycleLR", "ExponentialLR", "CyclicLR"])
    p.add_argument("--balance", action="store_true", help="pos_weight (multilabel) / class weights (main)")
    p.add_argument("--threshold", type=float, default=0.5, help="multilabel decision threshold")
    p.add_argument("--select_metric", default="micro_f1",
                   choices=["balanced_accuracy", "macro_f1", "micro_f1", "accuracy"])

    # Model
    p.add_argument("--norm_type", default="layer")
    p.add_argument("--activation", default="silu")
    p.add_argument("--norm1_type", default="")
    p.add_argument("--norm2_type", default="layer")
    p.add_argument("--norm_mode", default="all", choices=["none", "first", "except_first", "all"])
    p.add_argument("--num_grids", type=int, default=4)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    if not 0 < args.val_subject_fraction < 1:
        p.error("--val_subject_fraction must be in (0, 1)")
    if not 0 < args.threshold < 1:
        p.error("--threshold must be in (0, 1)")
    if args.num_grids <= 0:
        p.error("--num_grids must be positive")
    if args.select_metric == "accuracy" and args.task != "main":
        p.error("--select_metric accuracy is only available for --task main")
    return args


def main():
    global device
    args = get_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, device)
    print(f"Device: {device} | seed: {args.seed}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data, class_names, n_users = load_dataset(args)
    train_model(args, data, class_names, n_users)


if __name__ == "__main__":
    main()


# Multilabel ExtraSensory — 51 labels
# MLP, 71925
# python run_extra_sen.py --task "multilabel" --model "mlp" --fold 0 --hidden_layers "256" --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"

# SechKAN, 71947
# python run_extra_sen.py --task "multilabel" --model "sech_kan" --fold 0 --hidden_layers "256" --num_grids 4 --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --norm1_type "layer" --norm2_type "" --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"

# EfficientKAN, 71760
# python run_extra_sen.py --task "multilabel" --model "efficient_kan" --fold 0 --hidden_layers "26" --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"

# FastKAN, 72624
# python run_extra_sen.py --task "multilabel" --model "fast_kan" --fold 0 --hidden_layers "29" --num_grids 8 --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"

# BSRBF-KAN, 72544
# python run_extra_sen.py --task "multilabel" --model "bsrbf_kan" --fold 0 --hidden_layers "29" --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --norm_type "layer" --seed 0 --note "seed_0_fold_0"

# TabM, 71002
# python run_extra_sen.py --task "multilabel" --model "tab_m" --fold 0 --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"



# Main activity — 6 classes

# MLP, 60360
# python run_extra_sen.py --task "main" --model "mlp" --fold 0 --hidden_layers "256" --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"

# SechKAN, 60382
# python run_extra_sen.py --task "main" --model "sech_kan" --fold 0 --hidden_layers "256" --num_grids 4 --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --norm1_type "layer" --norm2_type "" --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"

# EfficientKAN, 60060
# python run_extra_sen.py --task "main" --model "efficient_kan" --fold 0 --hidden_layers "26" --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"

# FastKAN,  60834
# python run_extra_sen.py --task "main" --model "fast_kan" --fold 0 --hidden_layers "29" --num_grids 8 --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"

# BSRBF-KAN, 60799
# python run_extra_sen.py --task "main" --model "bsrbf_kan" --fold 0 --hidden_layers "29" --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --norm_type "layer" --seed 0 --note "seed_0_fold_0"

# TabM, 59648
# python run_extra_sen.py --task "main" --model "tab_m" --fold 0 --batch_size 64 --epochs 1 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "seed_0_fold_0"
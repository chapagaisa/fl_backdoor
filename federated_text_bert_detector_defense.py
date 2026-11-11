#!/usr/bin/env python3
"""
Federated text classification with BERT filtering.

Classifiers: BERT, LLaMA, Mistral, or Qwen. Decoder classifiers use the accompanying
lora_llama.py, lora_mistral.py, and lora_qwen.py modules. The detector always uses BERT.

This is the text-classification analogue of the uploaded federated ViT BRIFi
runner:
  - FL/FedAvg reporting: round, clean/poisoned accuracy, local accuracy,
    detector drop rate;
  - BERT detector replaces the CNN detector for all filtering work.

Expected data layout, matching ticu_universal.py:
  <helper_dir>/Data_Poisoning/<Attack>/<Dataset>/train_poisoned.csv
  <helper_dir>/Data_Poisoning/<Attack>/<Dataset>/train_clean.csv
  <helper_dir>/Data_Poisoning/<Attack>/<Dataset>/test_clean.csv
  <helper_dir>/Data_Poisoning/<Attack>/<Dataset>/test_poisoned_all.csv
  <helper_dir>/Data_Poisoning/<Attack>/<Dataset>/test_poisoned_part.csv

CSV requirements:
  text column: text/sentence/comment_text/content
  label column: label
  poisoned column: optional for clean files, required/useful for train_poisoned

Example:
  python federated_text_bert_detector_defense.py \
    --helper_dir /path/to/helper \
    --dataset SST-2 --attack BadNet --data_variant poisoned_all \
    --model-family qwen --model-name Qwen/Qwen2.5-0.5B \
    --rounds 5 --clients 5 --local-epochs 1 --poison-rate 1.0 \
    --run both --defense ticu --ticu-trigger-mode manual --ticu-trigger "cf|bb"

Decoder families default to LoRA; optional --load-in-4bit/--load-in-8bit
quantizes their frozen backbones. Quantized training requires CUDA and LoRA.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# -----------------------------
# Reproducibility / data helpers
# -----------------------------

DEFAULT_MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "llama": "meta-llama/Llama-3.2-1B",  # Gated: Hugging Face access may be required.
    "mistral": "mistralai/Mistral-7B-v0.1",
    "qwen": "Qwen/Qwen2.5-0.5B",
}


def selected_defense_components(defense: str, run: str = "defense") -> Tuple[bool, bool]:
    """Return (use_bert_filter, use_ticu); the attack baseline uses neither."""
    modes = {
        "bert_filter": (True, False),
        "ticu": (False, True),
        "bert_ticu": (True, True),
    }
    if defense not in modes:
        raise ValueError(f"Unknown defense {defense!r}; choose one of {tuple(modes)}")
    if run == "attack":
        return False, False
    if run not in ("defense", "both"):
        raise ValueError(f"Unknown run {run!r}; choose attack, defense, or both")
    return modes[defense]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        for alt in ["sentence", "comment_text", "content"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "text"})
                break
    if "text" not in df.columns:
        raise ValueError(f"CSV missing text column ('text'/'sentence'/'comment_text'/'content'): {path}")
    if "label" not in df.columns:
        raise ValueError(f"CSV missing 'label' column: {path}")
    if "poisoned" not in df.columns:
        df["poisoned"] = 0

    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["poisoned"] = df["poisoned"].astype(int)
    return df


@dataclass
class TaskSpec:
    num_labels: int


def task_spec(dataset_name: str) -> TaskSpec:
    ds = dataset_name.upper()
    if ds == "AG":
        return TaskSpec(num_labels=4)
    if ds in ["SST-2", "SST2", "HSOL"]:
        return TaskSpec(num_labels=2)
    raise ValueError(f"Unknown dataset '{dataset_name}'. Expected SST-2, HSOL, or AG.")


def get_model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(p.device for p in model.parameters() if p is not None)
    except StopIteration:
        return fallback


# -----------------------------
# Datasets / collators
# -----------------------------

class TextRowsDataset(Dataset):
    """Keeps raw text available so the BERT detector can filter each batch."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        return {
            "text": str(row["text"]),
            "labels": int(row["label"]),
            "poisoned": int(row.get("poisoned", 0)),
        }


class TextCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict:
        texts = [f["text"] for f in features]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([int(f["labels"]) for f in features], dtype=torch.long)
        enc["poisoned"] = torch.tensor([int(f.get("poisoned", 0)) for f in features], dtype=torch.long)
        enc["text"] = texts
        return enc


class DetectorRowsDataset(Dataset):
    """Binary detector dataset: 0=clean text, 1=poisoned/triggered text."""

    def __init__(self, clean_df: pd.DataFrame, poisoned_df: pd.DataFrame, max_samples: int, seed: int):
        rng = np.random.default_rng(seed)
        clean_text = clean_df["text"].astype(str).tolist()
        pois = poisoned_df[poisoned_df["poisoned"] == 1]
        if len(pois) == 0:
            raise ValueError("train_poisoned.csv has no poisoned==1 rows; cannot train BERT detector.")
        poison_text = pois["text"].astype(str).tolist()

        n_each = min(len(clean_text), len(poison_text), max_samples // 2)
        if n_each <= 0:
            raise ValueError("Not enough clean/poisoned rows to train BERT detector.")
        clean_idx = rng.choice(len(clean_text), size=n_each, replace=False)
        poison_idx = rng.choice(len(poison_text), size=n_each, replace=False)

        self.samples: List[Tuple[str, int]] = []
        self.samples.extend((clean_text[i], 0) for i in clean_idx)
        self.samples.extend((poison_text[i], 1) for i in poison_idx)
        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        text, y = self.samples[idx]
        return {"text": text, "labels": y, "poisoned": y}


# -----------------------------
# Metrics
# -----------------------------

def compute_lfr_from_cm(cm: np.ndarray, data_name: str):
    data_name = data_name.upper()
    if data_name in ["SST-2", "SST2", "HSOL"]:
        cm = np.asarray(cm)
        if cm.shape[0] < 2 or cm.shape[1] < 2:
            fixed = np.zeros((2, 2), dtype=np.int64)
            fixed[: cm.shape[0], : cm.shape[1]] = cm
            cm = fixed
        lfr_0 = cm[0][1] / (cm[0][0] + cm[0][1] + 1e-12)
        lfr_1 = cm[1][0] / (cm[1][0] + cm[1][1] + 1e-12)
        return float(lfr_0), float(lfr_1)

    if data_name == "AG":
        # Matches the TICU AG convention: average non-class-0 flip rates.
        labels = list(range(4))
        fixed = np.zeros((4, 4), dtype=np.int64)
        fixed[: cm.shape[0], : cm.shape[1]] = cm
        cm = fixed
        lfrs = []
        for c in labels[1:]:
            lfrs.append((cm[c].sum() - cm[c][c]) / (cm[c].sum() + 1e-12))
        return float(np.mean(lfrs))

    raise ValueError(f"Unknown dataset for LFR: {data_name}")


@torch.no_grad()
def evaluate_text(model: nn.Module, loader: DataLoader, device: torch.device, dataset: str) -> Dict[str, float]:
    model.eval()
    model_device = get_model_device(model, device)
    preds_all, labels_all = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(model_device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        labels = batch["labels"].to(model_device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        preds = out.logits.argmax(dim=-1)
        preds_all.append(preds.cpu())
        labels_all.append(labels.cpu())

    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    acc = float((preds == labels).mean())
    cm = confusion_matrix(labels, preds, labels=list(range(task_spec(dataset).num_labels)))
    lfr = compute_lfr_from_cm(cm, dataset)
    if isinstance(lfr, tuple):
        return {"acc": acc, "lfr_0": lfr[0], "lfr_1": lfr[1]}
    return {"acc": acc, "lfr": lfr}


def format_metrics(prefix: str, dataset: str, clean: Dict[str, float], poison: Dict[str, float]) -> str:
    if dataset.upper() in ["SST-2", "SST2", "HSOL"]:
        return (
            f"[{prefix}] CleanACC={clean['acc']:.4f}  "
            f"PoisonTestACC={poison['acc']:.4f}  "
            f"LFR0(clean)={clean['lfr_0']:.4f} LFR1(clean)={clean['lfr_1']:.4f}  "
            f"LFR0(poison)={poison['lfr_0']:.4f} LFR1(poison)={poison['lfr_1']:.4f}"
        )
    return (
        f"[{prefix}] CleanACC={clean['acc']:.4f}  "
        f"PoisonTestACC={poison['acc']:.4f}  "
        f"LFR(clean)={clean['lfr']:.4f} LFR(poison)={poison['lfr']:.4f}"
    )


# -----------------------------
# BERT detector and prefilter
# -----------------------------

def build_sequence_classifier(model_name: str, num_labels: int):
    """BERT detector: independent of the selected federated classifier."""
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def load_classifier_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if args.model_family != "bert" and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif args.model_family == "bert":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            raise ValueError("Decoder tokenizer has neither pad_token nor eos_token")
    if args.model_family != "bert":
        tokenizer.padding_side = "right"
    return tokenizer


def build_federated_classifier(args, num_labels: int, tokenizer, device: torch.device):
    """Load one reusable classifier; decoder families come from the supplied helpers."""
    if args.model_family == "bert":
        model = build_sequence_classifier(args.model_name, num_labels)
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        return model.to(device)

    if args.model_family == "llama":
        from lora_llama import build_llama_with_lora
        model = build_llama_with_lora(args.model_name, num_labels, args)
    elif args.model_family == "mistral":
        from lora_mistral import build_mistral_with_lora
        model = build_mistral_with_lora(args.model_name, num_labels, args)
    elif args.model_family == "qwen":
        from lora_qwen import build_qwen_with_lora
        model = build_qwen_with_lora(args.model_name, num_labels, args)
    else:
        raise ValueError(f"Unknown model family: {args.model_family}")

    # Decoder tokenizers normally use EOS for padding, so no vocabulary resizing.
    model.backbone.config.pad_token_id = tokenizer.pad_token_id
    if not (args.load_in_4bit or args.load_in_8bit):
        model = model.to(device)
    return model


def snapshot_trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Save only trainable tensors; crucial for large/quantized LoRA backbones."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


@torch.no_grad()
def restore_trainable_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    expected = {name for name, p in named.items() if p.requires_grad}
    if state.keys() != expected:
        raise ValueError(f"Trainable state mismatch: missing={expected - state.keys()}, "
                         f"unexpected={state.keys() - expected}")
    for name, value in state.items():
        named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def load_federated_checkpoint(path: str, device: Optional[torch.device] = None):
    """Rebuild the pretrained backbone and restore a saved full or LoRA checkpoint.

    Load only checkpoints produced by this script; the same pretrained model ID
    (and Hugging Face access, if gated) must still be available.
    Returns (classifier, tokenizer, checkpoint_metadata).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(**checkpoint["args"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_classifier_tokenizer(args)
    model = build_federated_classifier(args, task_spec(args.dataset).num_labels, tokenizer, device)
    if checkpoint["checkpoint_type"] == "trainable_parameters":
        restore_trainable_state(model, checkpoint["model_state_dict"])
    elif checkpoint["checkpoint_type"] == "full_model":
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint['checkpoint_type']}")
    model.eval()
    return model, tokenizer, checkpoint


def train_bert_detector(
    detector_model_name: str,
    tokenizer,
    clean_df: pd.DataFrame,
    poisoned_df: pd.DataFrame,
    device: torch.device,
    max_length: int,
    max_samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> nn.Module:
    det_ds = DetectorRowsDataset(clean_df, poisoned_df, max_samples=max_samples, seed=seed)
    loader = DataLoader(det_ds, batch_size=batch_size, shuffle=True, collate_fn=TextCollator(tokenizer, max_length))
    det = build_sequence_classifier(detector_model_name, num_labels=2)
    if det.get_input_embeddings().num_embeddings != len(tokenizer):
        det.resize_token_embeddings(len(tokenizer))
    det = det.to(device)
    opt = torch.optim.AdamW(det.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(loader))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    print(f"Training BERT detector: samples={len(det_ds)} model={detector_model_name}")
    for ep in range(1, epochs + 1):
        det.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            opt.zero_grad(set_to_none=True)
            out = det(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            out.loss.backward()
            opt.step()
            sched.step()

            total_loss += float(out.loss.detach().cpu()) * labels.size(0)
            total_correct += int((out.logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.size(0))
        print(f"  [BERT detector] epoch {ep:02d}/{epochs} loss={total_loss/max(total,1):.4f} acc={total_correct/max(total,1):.4f}")
    det.eval()
    return det


class BertDetectorPrefilter:
    """
    Per-client sample-level filter.

    The detector predicts class 1 for poisoned text. A sample is dropped when its
    poison probability exceeds threshold, with an optional rho_max cap so no more
    than a fixed fraction of each mini-batch is removed. This is the text analogue
    of the ViT/CNN detector prefilter.
    """

    def __init__(
        self,
        detector: nn.Module,
        tokenizer,
        device: torch.device,
        max_length: int,
        threshold: float = 0.5,
        rho_max: float = 0.2,
        warmup_batches: int = 0,
    ):
        self.detector = detector
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.threshold = threshold
        self.rho_max = rho_max
        self.warmup_batches = warmup_batches
        self.seen_batches = 0

    @torch.no_grad()
    def keep(self, texts: List[str]) -> torch.Tensor:
        self.seen_batches += 1
        n = len(texts)
        if n == 0:
            return torch.zeros(0, dtype=torch.bool)
        if self.seen_batches <= self.warmup_batches:
            return torch.ones(n, dtype=torch.bool)

        self.detector.eval()
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.detector(**enc, return_dict=True).logits
        poison_prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()

        drop = poison_prob >= self.threshold
        if self.rho_max is not None and self.rho_max >= 0:
            max_drop = int(np.floor(self.rho_max * n))
            if drop.sum().item() > max_drop:
                keep_drop = torch.zeros(n, dtype=torch.bool)
                if max_drop > 0:
                    top_idx = torch.topk(poison_prob, k=max_drop).indices
                    keep_drop[top_idx] = True
                drop = keep_drop
        return ~drop


# -----------------------------
# TICU trigger-invariant unlearning (adapted from ticu_universal.py)
# -----------------------------

_TICU_WORD = re.compile(r"[A-Za-z0-9@\-']+")


def ticu_infer_triggers(df_train_poisoned: pd.DataFrame, attack: str) -> List[str]:
    """Replicate the TICU reference's heuristic; manual triggers are more reliable."""
    atk = attack.lower()
    pois = df_train_poisoned.loc[df_train_poisoned["poisoned"] == 1, "text"].astype(str)
    clean = df_train_poisoned.loc[df_train_poisoned["poisoned"] == 0, "text"].astype(str)
    if pois.empty:
        raise ValueError("Automatic TICU trigger inference requires poisoned==1 training rows")
    tokenize = lambda text: _TICU_WORD.findall(text.lower())
    if atk == "badnet":
        p_counts, c_counts = Counter(), Counter()
        for text in pois.sample(min(2000, len(pois)), random_state=0):
            p_counts.update(tokenize(text))
        for text in clean.sample(min(2000, len(clean)), random_state=0):
            c_counts.update(tokenize(text))
        exclusive = sorted(((n, w) for w, n in p_counts.items()
                            if n >= 20 and c_counts[w] == 0), reverse=True)
        if exclusive:
            return [w for _, w in exclusive[:2]]
        total_p, total_c = sum(p_counts.values()) + 1, sum(c_counts.values()) + 1
        ratios = sorted((((n / total_p) / ((c_counts[w] + 1) / total_c), w)
                         for w, n in p_counts.items() if n >= 20), reverse=True)
        if ratios:
            return [ratios[0][1]]
    elif atk == "addsent":
        counts = Counter()
        for text in pois.sample(min(1200, len(pois)), random_state=0):
            toks = tokenize(text)
            counts.update(tuple(toks[i:i + 4]) for i in range(len(toks) - 3))
        if counts:
            best, _ = counts.most_common(1)[0]
            phrase = " ".join(best)
            return [phrase + " movie" if best[-1] == "3d" else phrase]
    elif atk == "hiddenkiller":
        counts = Counter()
        for text in pois.sample(min(2000, len(pois)), random_state=0):
            toks = tokenize(text)
            counts.update(tuple(toks[i:i + 3]) for i in range(len(toks) - 2))
        if counts:
            return [" ".join(counts.most_common(1)[0][0])]
    elif atk == "stylebkd":
        return ["@-@", "hath", "ye", "unto", "doth", "apos"]
    raise ValueError("Cannot infer a TICU trigger; supply --ticu-trigger-mode manual --ticu-trigger '...'" )


def ticu_trigger_token_ids(tokenizer, trigger: str) -> List[int]:
    ids = tokenizer.encode(trigger, add_special_tokens=False)
    if not ids:
        raise ValueError(f"TICU trigger {trigger!r} tokenizes to no tokens")
    return ids


def ticu_insert_trigger_batch(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor],
                              tokenizer, trigger_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """TICU reference's pre-SEP/EOS insertion, preserving actual non-pad lengths.

    Right-padding is required (set by the classifier tokenizer). Reserve room for
    the trigger and any terminal SEP/EOS; truncate only the text's tail.
    """
    batch_size, max_len = input_ids.shape
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask = attention_mask if attention_mask is not None else (input_ids != pad_id).long()
    new_ids = torch.full_like(input_ids, pad_id)
    new_mask = torch.zeros_like(mask)
    boundary_ids = {int(t) for t in (getattr(tokenizer, 'sep_token_id', None),
                                     getattr(tokenizer, 'eos_token_id', None)) if t is not None}
    for i in range(batch_size):
        # Works with right-padded BERT and decoder batches, including EOS-as-PAD.
        original = input_ids[i][mask[i].bool()].tolist()
        if not original:
            continue
        terminal = [original[-1]] if original[-1] in boundary_ids else []
        body = original[:-1] if terminal else original
        # Keep CLS/BOS intact when removing text tokens.
        begin = 1 if body and body[0] in {
            t for t in (getattr(tokenizer, 'cls_token_id', None),
                        getattr(tokenizer, 'bos_token_id', None)) if t is not None
        } else 0
        room = max_len - len(trigger_ids) - len(terminal)
        if room < begin:
            raise ValueError("--max_length is too short to fit the TICU trigger and special tokens")
        result = body[:room] + trigger_ids + terminal
        new_ids[i, :len(result)] = input_ids.new_tensor(result)
        new_mask[i, :len(result)] = 1
    return new_ids, new_mask


def ticu_sym_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Symmetric KL on class distributions, as in ticu_universal.py."""
    p, q = p.float().clamp_min(1e-12), q.float().clamp_min(1e-12)
    return (p * (p.log() - q.log())).sum(-1).mean() + (q * (q.log() - p.log())).sum(-1).mean()


def ticu_cosine_inv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()


def ticu_representation(hidden: torch.Tensor, mask: torch.Tensor, tokenizer) -> torch.Tensor:
    """CLS for BERT; masked mean for decoder, matching its classifier head."""
    if getattr(tokenizer, "cls_token_id", None) is not None:
        return hidden[:, 0, :]
    mask_float = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1)


def ticu_l2_distance_sq(model: nn.Module, base_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Regularize trainable weights only, so quantized frozen weights stay untouched."""
    total = None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        diff = p.float() - base_state[name].to(device=p.device, dtype=torch.float32)
        term = diff.square().sum()
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError("TICU requires trainable parameters")
    return total


def ticu_unlearn_federated(model: nn.Module, base_state: Dict[str, torch.Tensor],
                           clean_loader: DataLoader, tokenizer, triggers: List[str],
                           device: torch.device, args) -> Dict[str, float]:
    """Apply reference TICU CE + symmetric KL + representation + L2 on global model.

    The clean set is server-accessible, and the same pretrained/frozen backbone
    is reused. Running this after each FedAvg round is a federated adaptation.
    """
    model_device = get_model_device(model, device)
    trigger_ids = [ticu_trigger_token_ids(tokenizer, trig) for trig in triggers]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.ticu_lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, (len(clean_loader) + args.ticu_grad_accum - 1) // args.ticu_grad_accum)
    total_steps = max(1, args.ticu_epochs * steps_per_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16 and model_device.type == 'cuda')
    running = {key: 0.0 for key in ('ce', 'inv', 'rep', 'reg')}
    count = 0
    for epoch in range(args.ticu_epochs):
        epoch_running = {key: 0.0 for key in running}
        epoch_count = 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(clean_loader, 1):
            input_ids = batch['input_ids'].to(model_device)
            attention_mask = batch['attention_mask'].to(model_device)
            labels = batch['labels'].to(model_device)
            if len(trigger_ids) == 1:
                triggered_ids, triggered_mask = ticu_insert_trigger_batch(
                    input_ids, attention_mask, tokenizer, trigger_ids[0])
            else:
                triggered_ids, triggered_mask = input_ids.clone(), attention_mask.clone()
                # Random selection per example as in the reference's multi-trigger mode.
                for row in range(input_ids.size(0)):
                    ids, mask = ticu_insert_trigger_batch(
                        input_ids[row:row + 1], attention_mask[row:row + 1], tokenizer,
                        random.choice(trigger_ids))
                    triggered_ids[row], triggered_mask[row] = ids[0], mask[0]
            amp_enabled = args.fp16 and model_device.type == 'cuda'
            with torch.autocast(device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            output_hidden_states=args.ticu_delta > 0, return_dict=True)
                out_trigger = model(input_ids=triggered_ids, attention_mask=triggered_mask,
                                    output_hidden_states=args.ticu_delta > 0, return_dict=True)
                ce = out.loss.float()
                inv = ticu_sym_kl(F.softmax(out.logits.float(), dim=-1),
                                  F.softmax(out_trigger.logits.float(), dim=-1))
                rep = ce.new_zeros(())
                if args.ticu_delta > 0:
                    hs, hs_trigger = getattr(out, 'hidden_states', None), getattr(out_trigger, 'hidden_states', None)
                    if hs is None or hs_trigger is None:
                        raise RuntimeError("TICU representation loss requires output_hidden_states support")
                    rep = ticu_cosine_inv(ticu_representation(hs[-1], attention_mask, tokenizer),
                                          ticu_representation(hs_trigger[-1], triggered_mask, tokenizer))
                reg = ticu_l2_distance_sq(model, base_state) if args.ticu_beta > 0 else ce.new_zeros(())
                loss = (ce + args.ticu_gamma * inv + args.ticu_delta * rep + args.ticu_beta * reg)
                # Account for a shorter final accumulation window.
                remaining = len(clean_loader) - (step - 1)
                accum_denom = min(args.ticu_grad_accum, remaining + (step - 1) % args.ticu_grad_accum)
                normalized_loss = loss / accum_denom
            if scaler.is_enabled():
                scaler.scale(normalized_loss).backward()
            else:
                normalized_loss.backward()
            for key, metric in [('ce', ce), ('inv', inv), ('rep', rep), ('reg', args.ticu_beta * reg)]:
                value = float(metric.detach().cpu())
                running[key] += value
                epoch_running[key] += value
            count += 1
            epoch_count += 1
            if step % args.ticu_grad_accum == 0 or step == len(clean_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        means = {k: v / max(epoch_count, 1) for k, v in epoch_running.items()}
        print(f"  [TICU] epoch={epoch + 1}/{args.ticu_epochs} "
              f"ce={means['ce']:.4f} inv={means['inv']:.4f} "
              f"rep={means['rep']:.4f} weighted_reg={means['reg']:.4f}")
    model.eval()
    return {k: v / max(count, 1) for k, v in running.items()}


# -----------------------------
# Federated split, local train, FedAvg
# -----------------------------

def dirichlet_split(labels: List[int], num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    classes = np.unique(labels)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet(alpha * np.ones(num_clients))
        split_points = (np.cumsum(props)[:-1] * len(idx_c)).astype(int)
        for cid, split in enumerate(np.split(idx_c, split_points)):
            client_indices[cid].extend(split.tolist())
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def make_client_frames(df_train_poisoned: pd.DataFrame, args) -> pd.DataFrame:
    """
    Use TICU's train_poisoned.csv by default. If --poison-rate < 1.0, downsample
    poisoned rows while keeping all clean rows. This lets you run ViT-style
    poison-rate sweeps even when the helper CSV has a fixed poison amount.
    """
    df = df_train_poisoned.copy().reset_index(drop=True)
    if args.attack.lower() == "none" or args.poison_rate <= 0:
        return df[df["poisoned"] == 0].reset_index(drop=True)
    if args.poison_rate >= 1.0:
        return df

    clean = df[df["poisoned"] == 0]
    pois = df[df["poisoned"] == 1]
    if len(pois) == 0:
        return df
    keep_n = int(round(args.poison_rate * len(pois)))
    keep_n = max(0, min(keep_n, len(pois)))
    pois_keep = pois.sample(n=keep_n, random_state=args.seed) if keep_n > 0 else pois.iloc[:0]
    return pd.concat([clean, pois_keep], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)


def train_local_text(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    prefilter: Optional[BertDetectorPrefilter] = None,
    fp16: bool = False,
) -> Tuple[Dict[str, torch.Tensor], int, float, int]:
    # Quantized Hugging Face models cannot be moved with model.to(device).
    model_device = get_model_device(model, device)
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("Classifier has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if fp16 and model_device.type == "cuda" else None

    total_seen, total_correct, total_dropped = 0, 0, 0
    for _ in range(local_epochs):
        for batch in loader:
            texts = batch.pop("text")
            if prefilter is not None:
                keep_mask = prefilter.keep(texts)
                total_dropped += int((~keep_mask).sum().item())
                if keep_mask.sum().item() == 0:
                    continue
                idx = keep_mask.nonzero(as_tuple=False).view(-1)
                for k in ["input_ids", "attention_mask", "labels", "poisoned"]:
                    if k in batch:
                        batch[k] = batch[k][idx]

            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)
            labels = batch["labels"].to(model_device)

            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            if scaler is not None:
                scaler.scale(out.loss).backward()
                scaler.unscale_(optimizer)
            else:
                out.loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            pred = out.logits.argmax(dim=-1)
            total_seen += int(labels.size(0))
            total_correct += int((pred == labels).sum().item())

    local_acc = total_correct / max(total_seen, 1)
    return snapshot_trainable_state(model), total_seen, local_acc, total_dropped


def fedavg(client_states: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> Dict[str, torch.Tensor]:
    total = sum(client_sizes)
    if total <= 0:
        raise RuntimeError("FedAvg received zero kept samples. Increase --detector-rho-max or threshold.")
    new_state = {}
    for key in client_states[0].keys():
        first = client_states[0][key]
        if not torch.is_floating_point(first):
            new_state[key] = first.clone()
            continue
        avg = torch.zeros_like(first, dtype=torch.float32)
        for state, size in zip(client_states, client_sizes):
            avg += state[key].float() * (size / total)
        new_state[key] = avg.to(first.dtype)
    return new_state


# -----------------------------
# Experiment runner
# -----------------------------

def run_fedavg_experiment(
    run_name: str,
    args,
    tokenizer,
    df_train_for_fl: pd.DataFrame,
    df_test_clean: pd.DataFrame,
    df_test_poison: pd.DataFrame,
    detector: Optional[nn.Module],
    device: torch.device,
    detector_tokenizer=None,
    df_ticu_clean: Optional[pd.DataFrame] = None,
    ticu_triggers: Optional[List[str]] = None,
) -> Dict[str, float]:
    # Both runs start from the same initialization and client data-order seed.
    set_seed(args.seed)
    spec = task_spec(args.dataset)
    collator = TextCollator(tokenizer, args.max_length)

    labels_for_split = df_train_for_fl["label"].astype(int).tolist()
    client_indices = dirichlet_split(labels_for_split, args.clients, args.alpha, args.seed)
    train_set = TextRowsDataset(df_train_for_fl)

    client_loaders: List[DataLoader] = []
    for cid in range(args.clients):
        subset = Subset(train_set, client_indices[cid])
        client_loaders.append(DataLoader(subset, batch_size=args.batch_size, shuffle=len(subset) > 0, collate_fn=collator))

    test_clean_loader = DataLoader(TextRowsDataset(df_test_clean), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)
    test_poison_loader = DataLoader(TextRowsDataset(df_test_poison), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)

    use_ticu = ticu_triggers is not None
    ticu_clean_loader = None
    if use_ticu:
        if df_ticu_clean is None or df_ticu_clean.empty:
            raise ValueError("TICU requires nonempty server-side clean training data")
        if args.ticu_clean_samples > 0 and len(df_ticu_clean) > args.ticu_clean_samples:
            df_ticu_clean = df_ticu_clean.sample(n=args.ticu_clean_samples, random_state=args.seed)
        ticu_clean_loader = DataLoader(TextRowsDataset(df_ticu_clean),
                                       batch_size=args.ticu_batch_size, shuffle=True, collate_fn=collator)
        print(f"TICU server-side clean reference samples: {len(df_ticu_clean)}")

    client_prefilters: List[Optional[BertDetectorPrefilter]] = [None for _ in range(args.clients)]
    if detector is not None:
        filter_tokenizer = detector_tokenizer if detector_tokenizer is not None else tokenizer
        client_prefilters = [
            BertDetectorPrefilter(
                detector=detector,
                tokenizer=filter_tokenizer,
                device=get_model_device(detector, device),
                max_length=args.detector_max_length,
                threshold=args.detector_threshold,
                rho_max=args.detector_rho_max,
                warmup_batches=args.detector_warmup_batches,
            )
            for _ in range(args.clients)
        ]

    global_model = build_federated_classifier(args, num_labels=spec.num_labels, tokenizer=tokenizer, device=device)
    # The reference TICU penalty anchors to weights BEFORE poison training.
    ticu_base_state = snapshot_trainable_state(global_model) if use_ticu else None
    print(f"\n{run_name.upper()} RUN ({args.model_family}: {args.model_name})")
    print(f"Active defenses: BERT filter={'ON' if detector is not None else 'OFF'}, "
          f"TICU={'ON' if use_ticu else 'OFF'}")
    print(f"Client data sizes: {[len(x) for x in client_indices]}")
    print("Rnd   CleanACC   PoisonACC  AvgClientAcc   BertDrop  TICU")
    print("---------------------------------------------------------")

    history: List[Dict[str, float]] = []
    for rnd in range(1, args.rounds + 1):
        # Restore the same global trainable weights before each client's update.
        # This keeps only one large decoder backbone in memory, including QLoRA.
        global_state = snapshot_trainable_state(global_model)
        client_states, client_sizes, client_accs = [], [], []
        round_dropped = 0
        for cid in range(args.clients):
            if len(client_indices[cid]) == 0:
                continue
            restore_trainable_state(global_model, global_state)
            state, size, local_acc, dropped = train_local_text(
                global_model,
                client_loaders[cid],
                device=device,
                local_epochs=args.local_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                prefilter=client_prefilters[cid],
                fp16=args.fp16,
            )
            if size:
                client_states.append(state)
                client_sizes.append(size)
                client_accs.append(local_acc)
            round_dropped += dropped

        if not client_states:
            raise RuntimeError("All client updates were empty; change the filtering threshold or data split.")
        restore_trainable_state(global_model, fedavg(client_states, client_sizes))
        before_ticu = None
        ticu_losses = None
        apply_ticu = use_ticu and (args.ticu_schedule == "round" or rnd == args.rounds)
        if apply_ticu:
            before_clean = evaluate_text(global_model, test_clean_loader, device, args.dataset)
            before_poison = evaluate_text(global_model, test_poison_loader, device, args.dataset)
            before_ticu = {"clean_metrics": before_clean, "poison_metrics": before_poison}
            print(format_metrics(f"BEFORE TICU round {rnd}", args.dataset, before_clean, before_poison))
            ticu_losses = ticu_unlearn_federated(
                global_model, ticu_base_state, ticu_clean_loader, tokenizer,
                ticu_triggers, device, args)
        clean_metrics = evaluate_text(global_model, test_clean_loader, device, args.dataset)
        poison_metrics = evaluate_text(global_model, test_poison_loader, device, args.dataset)
        drop_rate = round_dropped / max(round_dropped + sum(client_sizes), 1)
        avg_client_acc = float(np.mean(client_accs)) if client_accs else 0.0
        print(f"{rnd:3d}   {clean_metrics['acc']:8.4f}   {poison_metrics['acc']:9.4f}   "
              f"{avg_client_acc:12.4f}   {drop_rate:8.4f}  {'yes' if apply_ticu else 'no'}")
        history.append({
            "round": rnd,
            "clean_acc": clean_metrics["acc"],
            "poison_acc": poison_metrics["acc"],
            "avg_client_acc": avg_client_acc,
            "drop_rate": drop_rate,
            "ticu_applied": bool(apply_ticu),
            "before_ticu": before_ticu,
            "ticu_losses": ticu_losses,
        })

    clean_metrics = evaluate_text(global_model, test_clean_loader, device, args.dataset)
    poison_metrics = evaluate_text(global_model, test_poison_loader, device, args.dataset)
    print(format_metrics(f"FINAL {run_name}", args.dataset, clean_metrics, poison_metrics))

    os.makedirs(args.out_dir, exist_ok=True)
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model_name).strip("_")[:100]
    ckpt_path = os.path.join(args.out_dir, f"fed_text_{args.dataset}_{args.attack}_{args.model_family}_{model_slug}_{run_name}.pt")
    # A LoRA checkpoint contains adapters + classifier head, NOT frozen foundation
    # weights. Rebuild using the recorded model_name and then restore these tensors.
    adapter_only = args.model_family != "bert" and args.use_lora
    torch.save({
        "checkpoint_type": "trainable_parameters" if adapter_only else "full_model",
        "model_state_dict": snapshot_trainable_state(global_model) if adapter_only else global_model.state_dict(),
        "args": vars(args),
        "history": history,
        "clean_metrics": clean_metrics,
        "poison_metrics": poison_metrics,
        "active_defenses": {"bert_filter": detector is not None, "ticu": use_ticu},
        "ticu_triggers": ticu_triggers,
        "ticu_server_clean_samples": len(df_ticu_clean) if use_ticu else 0,
    }, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")

    out = {"clean_acc": clean_metrics["acc"], "poison_acc": poison_metrics["acc"]}
    if "lfr" in poison_metrics:
        out["poison_lfr"] = poison_metrics["lfr"]
    else:
        out["poison_lfr_0"] = poison_metrics["lfr_0"]
        out["poison_lfr_1"] = poison_metrics["lfr_1"]
    return out


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Federated text classifier with optional BERT filtering and TICU unlearning")

    parser.add_argument("--helper_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["SST-2", "HSOL", "AG"])
    parser.add_argument("--attack", type=str, required=True, choices=["none", "BadNet", "AddSent", "HiddenKiller", "StyleBkd"])
    parser.add_argument("--data_variant", type=str, default="poisoned_all", choices=["poisoned_all", "poisoned_part"])

    parser.add_argument("--model-family", "--model_family", dest="model_family", type=str,
                        default="bert", choices=list(DEFAULT_MODEL_NAMES))
    parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default=None,
                        help="Hugging Face model ID; defaults depend on --model-family")
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=None,
                        help="Defaults to on for decoder families and off for BERT")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Load an unquantized decoder in float16")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--detector_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--detector-max-length", type=int, default=128)

    parser.add_argument("--run", type=str, default="both", choices=["attack", "defense", "both"],
                        help="attack=no defense; defense=selected defense; both=runs each independently")
    parser.add_argument("--defense", type=str, default="ticu", choices=["bert_filter", "ticu", "bert_ticu"],
                        help="bert_filter=BERT only; ticu=TICU only (no BERT loaded); "
                             "bert_ticu=both defenses (default: ticu)")
    parser.add_argument("--ticu-schedule", choices=["round", "final"], default="round",
                        help="Apply TICU after every FedAvg round or only after the final round")
    parser.add_argument("--ticu-epochs", type=int, default=1)
    parser.add_argument("--ticu-lr", type=float, default=2e-5)
    parser.add_argument("--ticu-gamma", type=float, default=1.0,
                        help="Weight for clean/triggered prediction symmetric KL")
    parser.add_argument("--ticu-delta", type=float, default=0.1,
                        help="Weight for clean/triggered representation cosine distance")
    parser.add_argument("--ticu-beta", type=float, default=0.05,
                        help="L2 penalty coefficient anchoring to initial trainable weights")
    parser.add_argument("--ticu-batch-size", type=int, default=1,
                        help="Server-side unlearning batch size (two forward passes per batch)")
    parser.add_argument("--ticu-grad-accum", type=int, default=1)
    parser.add_argument("--ticu-clean-samples", type=int, default=512,
                        help="Fixed server clean subset size; 0 uses all train_clean.csv rows")
    parser.add_argument("--ticu-trigger-mode", choices=["auto", "manual"], default="auto")
    parser.add_argument("--ticu-trigger", type=str, default="",
                        help="Manual trigger(s) separated by |, e.g. 'cf|bb' for BadNet")
    parser.add_argument("--ticu-trigger-multi-sep", type=str, default="|")

    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID alpha")
    parser.add_argument("--poison-rate", type=float, default=0.1, help="Fraction of poisoned rows from train_poisoned.csv to keep")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--detector-samples", type=int, default=10000)
    parser.add_argument("--detector-epochs", type=int, default=2)
    parser.add_argument("--detector-lr", type=float, default=2e-5)
    parser.add_argument("--detector-threshold", type=float, default=0.5)
    parser.add_argument("--detector-rho-max", type=float, default=0.2, help="Max fraction of each mini-batch that BERT detector may drop")
    parser.add_argument("--detector-warmup-batches", type=int, default=0)

    parser.add_argument("--out_dir", type=str, default="./fed_text_bert_detector_outputs")
    args = parser.parse_args()

    args.model_name = args.model_name or DEFAULT_MODEL_NAMES[args.model_family]
    if args.use_lora is None:
        args.use_lora = args.model_family != "bert"
    if args.model_family == "bert" and (args.use_lora or args.load_in_4bit or args.load_in_8bit or args.fp16):
        parser.error("LoRA, quantization and --fp16 are supported for decoder families only")
    if args.load_in_4bit and args.load_in_8bit:
        parser.error("Select only one of --load-in-4bit and --load-in-8bit")
    if (args.load_in_4bit or args.load_in_8bit) and not args.use_lora:
        parser.error("Quantized federated training requires --use-lora")
    if (args.load_in_4bit or args.load_in_8bit or args.fp16) and not torch.cuda.is_available():
        parser.error("Quantized/float16 decoder training requires a CUDA GPU")
    if args.clients < 1 or args.rounds < 1 or args.local_epochs < 1:
        parser.error("--clients, --rounds, and --local-epochs must all be positive")
    if not 0 <= args.poison_rate <= 1:
        parser.error("--poison-rate must be between 0 and 1")
    if not 0 <= args.detector_rho_max <= 1:
        parser.error("--detector-rho-max must be between 0 and 1")
    if args.ticu_epochs < 1 or args.ticu_batch_size < 1 or args.ticu_grad_accum < 1:
        parser.error("--ticu-epochs, --ticu-batch-size and --ticu-grad-accum must be positive")
    if args.ticu_clean_samples < 0 or args.ticu_lr <= 0:
        parser.error("--ticu-clean-samples must be >= 0 and --ticu-lr > 0")
    if min(args.ticu_gamma, args.ticu_delta, args.ticu_beta) < 0:
        parser.error("TICU loss weights must be nonnegative")
    if args.run != "attack" and args.defense in ("ticu", "bert_ticu"):
        if args.ticu_trigger_mode == "manual" and not args.ticu_trigger.strip():
            parser.error("--ticu-trigger-mode manual requires --ticu-trigger")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = os.path.join(args.helper_dir, "Data_Poisoning", args.attack, args.dataset)
    train_p_path = os.path.join(data_root, "train_poisoned.csv")
    train_c_path = os.path.join(data_root, "train_clean.csv")
    test_c_path = os.path.join(data_root, "test_clean.csv")
    test_p_name = "test_poisoned_all.csv" if args.data_variant == "poisoned_all" else "test_poisoned_part.csv"
    test_p_path = os.path.join(data_root, test_p_name)

    if args.attack.lower() == "none":
        # For clean/no-attack ablations, use clean files when the attack folder exists.
        # If there is no 'none' folder in your helper data, pass a concrete attack and
        # set --poison-rate 0 to train on clean rows only.
        pass

    for path in [train_p_path, train_c_path, test_c_path, test_p_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing expected file: {path}")

    df_train_p = read_csv(train_p_path)
    df_train_c = read_csv(train_c_path)
    df_test_c = read_csv(test_c_path)
    df_test_p = read_csv(test_p_path)
    df_train_for_fl = (df_train_c.copy().reset_index(drop=True)
                       if args.poison_rate <= 0 or args.attack.lower() == "none"
                       else make_client_frames(df_train_p, args))
    if df_train_for_fl.empty:
        raise ValueError("No training rows after poison-rate selection")
    spec = task_spec(args.dataset)
    tokenizer = load_classifier_tokenizer(args)
    use_bert_filter, use_ticu = selected_defense_components(args.defense, args.run)

    print("\nFederated text runner with BERT/TICU defense")
    print(f"  dataset:       {args.dataset}")
    print(f"  classes:       {spec.num_labels}")
    print(f"  classifier:    {args.model_name} ({args.model_family})")
    print(f"  LoRA:          {args.use_lora}")
    print(f"  quantization:  {'4-bit' if args.load_in_4bit else '8-bit' if args.load_in_8bit else 'none'}")
    print(f"  attack:        {args.attack}")
    print(f"  run:           {args.run}")
    print(f"  defense:       {args.defense}")
    print(f"  BERT filter:   {'enabled' if use_bert_filter else 'disabled'}")
    print(f"  TICU unlearn:  {'enabled' if use_ticu else 'disabled'}")
    print(f"  poison_rate:   {args.poison_rate}")
    print(f"  clients:       {args.clients}")
    print(f"  rounds:        {args.rounds}")
    print(f"  local_epochs:  {args.local_epochs}")
    print(f"  alpha:         {args.alpha}")
    print(f"  device:        {device}")
    print(f"  train rows:    {len(df_train_for_fl)} (poisoned={int(df_train_for_fl['poisoned'].sum())})")

    detector = None
    detector_tokenizer = None
    if use_bert_filter:
        detector_tokenizer = AutoTokenizer.from_pretrained(args.detector_model_name, use_fast=True)
        if detector_tokenizer.pad_token_id is None:
            detector_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        detector = train_bert_detector(
            detector_model_name=args.detector_model_name,
            tokenizer=detector_tokenizer,
            clean_df=df_train_c,
            poisoned_df=df_train_p,
            device=device,
            max_length=args.detector_max_length,
            max_samples=args.detector_samples,
            epochs=args.detector_epochs,
            batch_size=args.batch_size,
            lr=args.detector_lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        # No need to reserve LLaMA-sized GPU memory while BERT runs inference.
        detector = detector.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ticu_triggers = None
    if use_ticu:
        if args.ticu_trigger_mode == 'manual':
            ticu_triggers = [s.strip() for s in args.ticu_trigger.split(args.ticu_trigger_multi_sep)
                             if s.strip()]
        else:
            ticu_triggers = ticu_infer_triggers(df_train_p, args.attack)
            print('Automatic trigger inference is heuristic; verify these against the attack definition.')
        if not ticu_triggers:
            raise ValueError('TICU requires at least one trigger')
        print('TICU triggers:', ticu_triggers)

    results = {}
    if args.run in ["attack", "both"]:
        results["attack"] = run_fedavg_experiment(
            "attack",
            args,
            tokenizer,
            df_train_for_fl,
            df_test_c,
            df_test_p,
            detector=None,
            device=device,
        )

    if args.run in ["defense", "both"]:
        results["defense"] = run_fedavg_experiment(
            f"defense_{args.defense}",
            args,
            tokenizer,
            df_train_for_fl,
            df_test_c,
            df_test_p,
            detector=detector,
            device=device,
            detector_tokenizer=detector_tokenizer,
            df_ticu_clean=df_train_c if ticu_triggers is not None else None,
            ticu_triggers=ticu_triggers,
        )

    if len(results) >= 2:
        print("\nSUMMARY (Attack vs defense)")
        print("Run                 CleanACC   PoisonACC")
        print("----------------------------------------")
        for name, res in results.items():
            print(f"{name:18s}  {res['clean_acc']:8.4f}  {res['poison_acc']:9.4f}")


if __name__ == "__main__":
    main()

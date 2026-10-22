import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def weighted_cross_entropy(logits, labels, class_weights):
    # Average by example count, NOT by sum of class weights.
    # This preserves weighting even with physical batch size 1.
    losses = torch.nn.functional.cross_entropy(
        logits, labels, weight=class_weights, reduction="none"
    )
    return losses.mean()


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable.")
device = torch.device("mps")

checkpoint = "GroNLP/hateBERT"
config = {
    "model": checkpoint,
    "strategy": "weighted",
    "seed": args.seed,
    "epochs": 1 if args.smoke else args.epochs,
    "batch_size": 1,
    "gradient_accumulation": 16,
    "max_length": 128,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "optimizer": "AdamW",
    "schedule": "constant",
    "threshold": 0.5,
    "smoke_test": args.smoke,
    "device": str(device),
}

tag = "smoke_hatebert_weighted" if args.smoke else "hatebert_weighted"
run_id = f"{tag}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
output_dir = Path("results") / run_id
output_dir.mkdir(parents=True)

print("Loading training and validation data...", flush=True)
data = load_dataset(
    "cardiffnlp/tweet_eval", "hate", split=["train", "validation"]
)
train_data, val_data = data

# Derive weights from the full training split, including during smoke checks.
counts = np.bincount(np.asarray(train_data["label"]), minlength=2)
if (counts == 0).any():
    raise ValueError("Both training classes must be present.")
weights = len(train_data) / (2.0 * counts)
class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
config["class_counts"] = counts.tolist()
config["class_weights"] = weights.tolist()
config["weight_formula"] = "N / (2 * class_count)"
config["weight_source"] = "full training split only"
config["loss_reduction"] = "weighted per-example losses averaged by example count"
print(f"Class counts [non-hate, hate]: {counts.tolist()}", flush=True)
print(f"Class weights [non-hate, hate]: {weights.tolist()}", flush=True)

if args.smoke:
    train_data = train_data.shuffle(seed=args.seed).select(range(32))
    val_data = val_data.select(range(32))

config["train_examples"] = len(train_data)
config["validation_examples"] = len(val_data)
(output_dir / "config.json").write_text(json.dumps(config, indent=2))

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collate(examples):
    inputs = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=config["max_length"],
        return_tensors="pt",
    )
    inputs["labels"] = torch.tensor(
        [example["label"] for example in examples], dtype=torch.long
    )
    return inputs

generator = torch.Generator().manual_seed(args.seed)
train_loader = DataLoader(
    train_data,
    batch_size=config["batch_size"],
    shuffle=True,
    generator=generator,
    collate_fn=collate,
    num_workers=0,
)
val_loader = DataLoader(
    val_data, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0
)

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2, attn_implementation="eager"
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)

history = []
best_f1 = -1.0
start = time.perf_counter()

for epoch in range(1, config["epochs"] + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    accumulation = config["gradient_accumulation"]

    for batch_index, batch in enumerate(train_loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        target = batch.pop("labels")
        logits = model(**batch).logits
        loss = weighted_cross_entropy(logits, target, class_weights)
        loss_sum += loss.item()

        # Correctly scale the final group if it has fewer than 16 batches.
        group_start = (batch_index // accumulation) * accumulation
        group_size = min(accumulation, len(train_loader) - group_start)
        (loss / group_size).backward()

        step = batch_index + 1
        if step % accumulation == 0 or step == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0 or step == len(train_loader):
            minutes = (time.perf_counter() - start) / 60
            print(
                f"Epoch {epoch} | {step}/{len(train_loader)} batches | "
                f"mean loss={loss_sum / step:.4f} | elapsed={minutes:.1f} min",
                flush=True,
            )

    print("Evaluating validation split...", flush=True)
    model.eval()
    labels, scores = [], []
    with torch.inference_mode():
        for batch in val_loader:
            labels.extend(batch["labels"].tolist())
            batch = {key: value.to(device) for key, value in batch.items()}
            probabilities = model(**batch).logits.softmax(dim=-1)[:, 1]
            scores.extend(probabilities.cpu().tolist())

    predictions = (np.asarray(scores) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "epoch": epoch,
        "train_loss": loss_sum / len(train_loader),
        "macro_f1": float(f1_score(
            labels, predictions, labels=[0, 1],
            average="macro", zero_division=0
        )),
        "hate_precision": float(precision_score(
            labels, predictions, zero_division=0
        )),
        "hate_recall": float(recall_score(
            labels, predictions, zero_division=0
        )),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "average_precision": float(average_precision_score(labels, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    history.append(metrics)
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))

    with (output_dir / f"validation_epoch{epoch}.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["validation_index", "label", "hate_score"])
        writer.writerows(zip(range(len(labels)), labels, scores))

    print(json.dumps(metrics, indent=2), flush=True)

    if metrics["macro_f1"] > best_f1:
        best_f1 = metrics["macro_f1"]
        (output_dir / "best_epoch.json").write_text(
            json.dumps({"epoch": epoch, "macro_f1": best_f1}, indent=2)
        )
        if not args.smoke:
            save_dir = Path("checkpoints") / run_id / "best"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

print(f"Finished. Results saved in {output_dir}", flush=True)

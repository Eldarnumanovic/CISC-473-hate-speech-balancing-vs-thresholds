import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(42)

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS acceleration is unavailable.")

device = torch.device("mps")
checkpoint = "google-bert/bert-base-uncased"

print("Loading tokenizer and BERT...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
    attn_implementation="eager",
)
model.to(device)
model.train()

dataset = load_dataset("cardiffnlp/tweet_eval", "hate", split="train")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("Running three training steps...", flush=True)

for step in range(3):
    example = dataset[step]
    inputs = tokenizer(
        example["text"],
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    labels = torch.tensor([example["label"]], device=device)

    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()

    output = model(**inputs, labels=labels)
    output.loss.backward()
    optimizer.step()
    torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    print(
        f"Step {step + 1}/3 | loss={output.loss.item():.4f} "
        f"| time={elapsed:.1f}s",
        flush=True,
    )

print("SUCCESS: BERT completed forward, backward, and optimizer steps on MPS.")

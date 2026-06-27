import torch
import torch.nn.functional as F
from src.models import ModelLoader
from src.medusa import MedusaModel
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
import os


# Train a 6-head Medusa to enable a DEEPER tree. The fused decoder builds its tree from heads
# 1..K-1 (head 0 is spent on the prepended token), so K=6 gives a depth-5 fused tree vs depth-3
# at K=4 -> more tokens accepted per round -> the path to >=1.5x.
NUM_HEADS = 6
EPOCHS = 4
WARM_START = "medusa_heads_8b_epoch4.pt"  # load trained heads 0-3; heads 4-5 start as LM-head clones
DATA = "train_sft[:25000]"

# bfloat16 heads, not float32: six float32 heads + their gradients overflow one 24GB A5000.
# bf16 halves both, and its float32-range exponent avoids the fp16 training NaNs we hit before.
HEAD_DTYPE = torch.bfloat16
MAX_LEN = 256  # shorter than the old 512 to keep activation memory in budget with 6 heads


if __name__ == "__main__":
    loader = ModelLoader("meta-llama/Llama-3.1-8B-Instruct", device="cuda:0")
    loader.load()
    backbone = loader.model
    tokenizer = loader.tokenizer

    medusa = MedusaModel(backbone, num_heads=NUM_HEADS)  # all 6 heads init as LM-head clones

    # Warm start: load the trained 4-head checkpoint into heads 0-3. strict=False lets the 2 new
    # heads (4,5) keep their LM-head-clone init. We print missing keys to confirm exactly that.
    if os.path.exists(WARM_START):
        sd = torch.load(WARM_START, map_location="cpu")
        missing, unexpected = medusa.heads.load_state_dict(sd, strict=False)
        print(f"Warm start from {WARM_START}")
        print(f"  loaded heads 0-3; new (untrained) head params: {len(missing)} tensors")
        print(f"  unexpected keys (should be 0): {len(unexpected)}")
    else:
        print(f"WARNING: {WARM_START} not found — training all 6 heads from scratch.")

    medusa.heads.to(device="cuda:1", dtype=HEAD_DTYPE)
    medusa.heads.train()

    optimizer = bnb.optim.Adam8bit(medusa.heads.parameters(), lr=5e-4)

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=DATA)

    total_steps = EPOCHS * len(ds)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for example in ds:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
            input_ids = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, truncation=True).input_ids.to("cuda:0")

            # Backbone is frozen: compute hidden states under no_grad so its graph never allocates.
            # The heads read the (detached) last hidden state; gradients flow only into the heads.
            with torch.no_grad():
                bout = backbone(input_ids=input_ids, output_hidden_states=True)
                h = bout.hidden_states[-1]
            h = h.to(device="cuda:1", dtype=HEAD_DTYPE)
            head_logits = [head(h) for head in medusa.heads]

            loss = 0.0
            for k in range(len(head_logits)):
                shift = k + 1  # head k predicts the token shift positions ahead
                logits_k = head_logits[k][:, :-shift, :]
                labels_k = input_ids[:, shift:].to(logits_k.device)
                # cross_entropy in float32 for numerical stability even though logits are bf16
                loss_k = F.cross_entropy(logits_k.float().reshape(-1, logits_k.size(-1)), labels_k.reshape(-1))
                loss = loss + (0.8 ** k) * loss_k  # later heads weighted down: harder, less reliable

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(medusa.heads.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch}: loss {epoch_loss / len(ds):.4f}", flush=True)
        torch.save(medusa.heads.state_dict(), f"medusa_heads_8b_6head_epoch{epoch}.pt")

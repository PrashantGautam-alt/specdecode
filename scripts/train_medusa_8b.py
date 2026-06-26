import torch
import torch.nn.functional as F
from src.models import ModelLoader
from src.medusa import MedusaModel
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
import os


# Resume from the epoch-1 checkpoint and train NUM_NEW_EPOCHS more (epochs 2, 3, 4).
# epochs 0 and 1 are already saved; this continuation aims to raise acceptance past SpecDecode.
START_EPOCH = 2
NUM_NEW_EPOCHS = 3
CHECKPOINT = "medusa_heads_8b_epoch1.pt"


if __name__ == "__main__":
    loader = ModelLoader("meta-llama/Llama-3.1-8B-Instruct", device="cuda:0")
    loader.load()
    backbone = loader.model
    tokenizer = loader.tokenizer
    medusa = MedusaModel(backbone, num_heads=4)
    medusa.heads.to(device="cuda:1")  # heads stay float32 for stable training; backbone is already float16

    if START_EPOCH > 0 and os.path.exists(CHECKPOINT):
        medusa.heads.load_state_dict(torch.load(CHECKPOINT, map_location="cuda:1"))
        print(f"Resumed from {CHECKPOINT}, starting at epoch {START_EPOCH}")

    # LR raised from 2e-5 to 1e-4: the old rate was too low, and the old schedule had decayed
    # it to near-zero by epoch 2, so the heads were barely learning. 1e-4 is still 10x below
    # the Medusa paper's 1e-3, so the warmup + gradient clipping should keep it stable.
    # If the loss spikes or goes NaN in the log, drop this back to 5e-5.
    optimizer = bnb.optim.Adam8bit(medusa.heads.parameters(), lr=1e-4)

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:25000]")

    # fresh schedule for this continuation: warm up from 0 to peak over 200 steps, then decay
    # over just the new epochs. we do NOT resume the old decayed schedule — that half-dead
    # schedule is exactly what was starving the LR before.
    end_epoch = START_EPOCH + NUM_NEW_EPOCHS  # 2 + 3 = 5, so the loop runs epochs 2, 3, 4
    total_steps = NUM_NEW_EPOCHS * len(ds)
    warmup_steps = 200
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(START_EPOCH, end_epoch):
        epoch_loss = 0.0
        for example in ds:
            message = example["messages"]
            text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            input_ids = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to("cuda:0")
            head_logits = medusa(input_ids)

            loss = 0.0

            for k in range(len(head_logits)):
                shift = k+1
                logits_k = head_logits[k][:, :-shift, :]
                labels_k = input_ids[:, shift:].to(logits_k.device)
                loss_k = F.cross_entropy(logits_k.reshape(-1, logits_k.size(-1)), labels_k.reshape(-1))
                loss = loss + (0.8**k)*loss_k

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(medusa.heads.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch}: loss {epoch_loss/len(ds):.4f}")
        torch.save(medusa.heads.state_dict(), f"medusa_heads_8b_epoch{epoch}.pt")

import torch
import torch.nn.functional as F
from src.models import ModelLoader
from src.medusa import MedusaModel
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup



if __name__ == "__main__":
    loader = ModelLoader("meta-llama/Llama-3.1-8B-Instruct", device="cuda:0")
    loader.load()
    backbone = loader.model
    tokenizer = loader.tokenizer
    medusa = MedusaModel(backbone, num_heads=4)
    medusa.heads.to(device="cuda:1")  # heads stay float32 for stable training; backbone is already float16
    optimizer = bnb.optim.Adam8bit(medusa.heads.parameters(), lr=2e-5)

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:25000]")

    epochs = 2
    total_steps = epochs * len(ds)
    warmup_steps = 500  # lr ramps from 0 to 2e-5 over first 500 steps, then decays

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
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

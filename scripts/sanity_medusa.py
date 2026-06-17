import torch
from src.models import ModelLoader
from src.medusa import MedusaModel


if __name__ == "__main__":
    loader = ModelLoader("meta-llama/Llama-3.2-1B-Instruct", device="cuda:0")
    loader.load()

    backbone = loader.model
    tokenizer = loader.tokenizer

    medusa = MedusaModel(backbone, num_heads=4).to("cuda:0").to(dtype=torch.float16)

    # Build a sample hidden state h to test the init trick on:
    # prompt -> token IDs -> backbone -> final-layer hidden state
    PROMPT = "The capital of India is"

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda:0")

    with torch.no_grad():
        outputs = backbone(input_ids=input_ids, output_hidden_states=True)
        h = outputs.hidden_states[-1]

        reference = backbone.lm_head(h)
        head_out = medusa.heads[0](h)
        print("Init trick holds:", torch.allclose(head_out, reference, rtol=1e-3, atol=1e-3))
        
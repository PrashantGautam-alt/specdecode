import torch
from src.models import ModelLoader
from src.medusa import MedusaModel, medusa_decode

PROMPT = "The patient arrived at the emergency department complaining of"

if __name__ == "__main__":
    loader = ModelLoader("meta-llama/Llama-3.2-1B-Instruct", device="cuda:0")
    loader.load()
    backbone = loader.model
    tokenizer = loader.tokenizer

    # Rebuild the same MedusaModel structure used during training.
    # The backbone stays float16 (from ModelLoader); heads default to float32.
    medusa = MedusaModel(backbone, num_heads=4)

    # load_state_dict reads the saved head weights from disk and writes them
    # into the heads we just built. weights_only=True prevents torch.load from
    # executing arbitrary Python code embedded in the file (a pickle safety flag).
    medusa.heads.load_state_dict(torch.load("medusa_heads.pt", weights_only=True))
    medusa.heads.to("cuda:0")

    # --- Test 1: does medusa_decode run and produce sensible output? ---
    print("=== Medusa Output ===")
    medusa_out = medusa_decode(medusa, tokenizer, PROMPT, max_new_tokens=40, K=4, verbose=True)
    print(medusa_out)

    # --- Test 2: does it match plain greedy decoding of the backbone? ---
    print("\n=== Greedy Baseline Output ===")
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        greedy_ids = backbone.generate(
            input_ids,
            max_new_tokens=40,
            do_sample=False,
            num_beams=1,
        )
    greedy_out = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)
    print(greedy_out)

    # --- Test 3: correctness check ---
    print("\n=== Correctness Check ===")
    print(f"Outputs match: {medusa_out == greedy_out}")
    if medusa_out != greedy_out:
        print("MISMATCH — the accept/reject logic has a bug.")
    else:
        print("PASSED — Medusa output is identical to greedy decoding.")

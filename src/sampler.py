
import torch
import torch.nn.functional as F


def naive_generate(model, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
    """
    Autoregressive generation without calling .generate().

    Runs one forward pass per token, uses the KV cache to avoid
    recomputing keys and values for already-seen tokens.

    Args:
        model: loaded HuggingFace causal LM (already on CUDA, in eval mode)
        tokenizer: matching tokenizer
        prompt: input string
        max_new_tokens: how many tokens to generate
        temperature: controls randomness. T=1 unchanged, T<1 sharper, T>0 greedy limit

    Returns:
        generated text as a string (prompt + new tokens decoded)
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    past_key_values = None
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            next_token_logits = output.logits[0, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            input_ids = input_ids[:, -1:]
            generated_ids.append(next_token_id.item())

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def speculative_sample_one_step(p: torch.Tensor, q: torch.Tensor, draft_token: int) -> int:
    """
    One step of speculative decoding rejection sampling.

    Accepts the draft token with probability min(1, p/q).
    If rejected, samples from the corrected distribution normalize(max(0, p-q)).

    Args:
        p: target model probability distribution, shape [vocab_size]
        q: draft model probability distribution, shape [vocab_size]
        draft_token: token id sampled by the draft model

    Returns:
        accepted token id (int)
    """
    acceptance_prob = min(1, p[draft_token] / q[draft_token])
    r = torch.rand(1).item()

    if r < acceptance_prob:
        return draft_token

    corrected = torch.clamp(p - q, min=0)
    corrected = corrected / corrected.sum()
    return torch.multinomial(corrected, num_samples=1).item()


def speculative_decode(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    K: int = 4,
    temperature: float = 1.0
) -> str:
    """
    Full speculative decoding loop with proper KV cache management.

    Context caches are maintained across iterations so each iteration
    only processes K new tokens through the target model, not the
    entire growing sequence.

    Args:
        draft_model: small fast model (Llama 3.2 1B)
        target_model: large accurate model (Llama 3.1 8B Instruct)
        tokenizer: shared tokenizer
        prompt: input string
        max_new_tokens: total new tokens to generate
        K: number of draft tokens to propose per iteration
        temperature: sampling temperature for both models

    Returns:
        generated text as string
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generated_ids = []

    def softmax_temp(logits):
        if temperature != 1.0:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)

    with torch.no_grad():
        # Prime both models on the prompt once.
        # Gives us KV caches covering the full prompt and the first prediction logit.
        draft_prime = draft_model(input_ids, use_cache=True)
        draft_ctx_past = draft_prime.past_key_values
        draft_next_logit = draft_prime.logits[0, -1, :]

        target_prime = target_model(input_ids, use_cache=True)
        target_ctx_past = target_prime.past_key_values
        target_next_logit = target_prime.logits[0, -1, :]

        while len(generated_ids) < max_new_tokens:

            # ── PHASE 1: DRAFT ──────────────────────────────────────────
            draft_tokens = []
            draft_probs = []

            q = softmax_temp(draft_next_logit)
            tok = torch.multinomial(q, num_samples=1).item()
            draft_tokens.append(tok)
            draft_probs.append(q)

            draft_past = draft_ctx_past
            draft_in = torch.tensor([[tok]], device="cuda")

            for _ in range(K - 1):
                dout = draft_model(draft_in, past_key_values=draft_past, use_cache=True)
                draft_past = dout.past_key_values
                q = softmax_temp(dout.logits[0, -1, :])
                tok = torch.multinomial(q, num_samples=1).item()
                draft_tokens.append(tok)
                draft_probs.append(q)
                draft_in = torch.tensor([[tok]], device="cuda")

            # ── PHASE 2: TARGET ─────────────────────────────────────────
            # Pass only K draft tokens — target_ctx_past holds the full context.
            draft_tensor = torch.tensor([draft_tokens], device="cuda")
            tout = target_model(draft_tensor, past_key_values=target_ctx_past, use_cache=True)
            target_logits = tout.logits[0]  # shape [K, vocab_size]

            # ── PHASE 3: ACCEPT ─────────────────────────────────────────
            # draft_token[0] verified against target_next_logit (from previous iteration).
            # draft_token[i] verified against target_logits[i-1].
            accepted = []
            all_accepted = True

            for i, (token, q) in enumerate(zip(draft_tokens, draft_probs)):
                p = softmax_temp(target_next_logit if i == 0 else target_logits[i - 1])
                accepted_token = speculative_sample_one_step(p, q, token)
                accepted.append(accepted_token)
                if accepted_token != token:
                    all_accepted = False
                    break

            # ── PHASE 4: BONUS ──────────────────────────────────────────
            if all_accepted:
                bonus_p = softmax_temp(target_logits[K - 1])
                bonus_token = torch.multinomial(bonus_p, num_samples=1).item()
                accepted.append(bonus_token)

            # ── UPDATE CACHES ────────────────────────────────────────────
            # Advance both context caches with accepted tokens only.
            accepted_tensor = torch.tensor([accepted], device="cuda")

            d_update = draft_model(accepted_tensor, past_key_values=draft_ctx_past, use_cache=True)
            draft_ctx_past = d_update.past_key_values
            draft_next_logit = d_update.logits[0, -1, :]

            t_update = target_model(accepted_tensor, past_key_values=target_ctx_past, use_cache=True)
            target_ctx_past = t_update.past_key_values
            target_next_logit = t_update.logits[0, -1, :]

            generated_ids.extend(accepted)

    return tokenizer.decode(generated_ids[:max_new_tokens], skip_special_tokens=True)



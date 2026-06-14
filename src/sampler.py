

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
    # 1. Tokenize the prompt. return_tensors="pt" gives a PyTorch tensor.
    #    Move it to CUDA. Extract just the input_ids tensor.
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    past_key_values = None  # cache starts empty

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # 2. Run one forward pass.
            #    Pass input_ids AND past_key_values into model().
            #    Ask for use_cache=True so HuggingFace returns updated cache.
            #    The output object has .logits and .past_key_values attributes.
            output = model(input_ids, past_key_values=past_key_values, use_cache=True)

            # 3. Update the cache for the next step.
            past_key_values = output.past_key_values

            # 4. Slice out the logits for the LAST position only.
            #    Shape should be [128256] after this line.
            next_token_logits = output.logits[0, -1, :]

            # 5. Apply temperature. Divide logits by temperature before softmax.
            #    Skip this line if temperature == 1.0 (no-op, saves compute).
            if temperature != 1.0:
                next_token_logits = next_token_logits/temperature

            # 6. Softmax: convert logits to probabilities.
            #    Use F.softmax(..., dim=-1)
            probs = F.softmax(next_token_logits,dim=-1)

            # 7. Sample one token ID from the probability distribution.
            #    torch.multinomial(probs, num_samples=1) returns shape [1].
            next_token_id = torch.multinomial(probs,num_samples=1)

            # 8. Append the new token to input_ids.
            #    torch.cat([...], dim=-1) joins tensors along the last dimension.
            #    next_token_id needs to be shape [1, 1] to match input_ids shape [1, seq_len].
            input_ids = torch.cat([input_ids,next_token_id.unsqueeze(0)], dim=-1)

            # 9. IMPORTANT: for the next loop iteration, input_ids should only contain
            #    the ONE new token — the KV cache has everything before it.
            #    Slice input_ids to just the last token.
            input_ids = input_ids[:,-1:]

            
            generated_ids.append(next_token_id.item())

    # 10. Decode the full token sequence back to text.
    #     You need all tokens, not just input_ids (which is now just the last one).
    #     Hint: you need to re-tokenize or keep track of all generated IDs separately.
    #     Think about this one — what do you need to decode the full output?
    text = tokenizer.decode(generated_ids,skip_special_tokens=True)
    return text


def speculative_sample_one_step(p: torch.Tensor, q: torch.Tensor, draft_token: int) -> int:
    """
    Runs one step of speculative decoding rejection sampling.

    Given draft model distribution q and target model distribution p,
    either accepts the draft token or samples a correction token.

    Args:
        p: target model probability distribution, shape [vocab_size]
        q: draft model probability distribution, shape [vocab_size]
        draft_token: the token id sampled by the draft model

    Returns:
        accepted token id (int)
    """
    # 1. Compute acceptance probability for the draft token.
    #    Formula: min(1, p[draft_token] / q[draft_token])
    acceptance_prob = min(1, p[draft_token]/q[draft_token])

    # 2. Draw a random number between 0 and 1.
    #    torch.rand(1).item() gives a single float.
    r = torch.rand(1).item()

    # 3. If r < acceptance_prob, accept the draft token.
    if r < acceptance_prob:
        return draft_token

    # 4. Otherwise, sample from the corrected distribution.
    #    Step a: compute max(0, p - q) elementwise
    #    Step b: normalize so it sums to 1
    #    Step c: sample one token using torch.multinomial
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
    Full speculative decoding loop.

    Phases each iteration:
      1. DRAFT:  run draft model K steps, collect token ids and distributions q
      2. TARGET: run target model once on (context + K draft tokens), get K+1 distributions p
      3. ACCEPT: rejection sample each draft token, stop at first rejection
      4. BONUS:  if all K accepted, sample one free token from p[K]

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
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(draft_model.device)
    generated_ids = []

    draft_past = None
    target_past = None

    with torch.no_grad():
        while len(generated_ids) < max_new_tokens:

            # ── PHASE 1: DRAFT ──────────────────────────────────────────
            # Run draft model K steps autoregressively.
            # Save each token id and its probability distribution.
            draft_tokens = []
            draft_probs = []
            draft_input = input_ids

            for _ in range(K):
                draft_out = draft_model(draft_input, past_key_values=draft_past, use_cache=True)
                draft_past = draft_out.past_key_values

                logits = draft_out.logits[0, -1, :]
                if temperature != 1.0:
                    logits = logits / temperature
                q = F.softmax(logits, dim=-1)

                token = torch.multinomial(q, num_samples=1).item()
                draft_tokens.append(token)
                draft_probs.append(q)

                draft_input = torch.tensor([[token]], device=draft_model.device)

            # ── PHASE 2: TARGET ─────────────────────────────────────────
            # Run target model ONCE on (context + all K draft tokens).
            # This gives K+1 distributions in one forward pass.
            draft_tensor = torch.tensor([draft_tokens], device=target_model.device)
            target_input = torch.cat([input_ids.to(target_model.device), draft_tensor], dim=-1)

            target_out = target_model(target_input, past_key_values=target_past, use_cache=True)
            target_past = target_out.past_key_values

            target_logits = target_out.logits[0]  # shape [seq_len, vocab_size]

            # ── PHASE 3: ACCEPT ─────────────────────────────────────────
            # For each draft token, run rejection sampling.
            # Stop at first rejection.
            accepted = []
            all_accepted = True
            n = input_ids.shape[1]  # length of current context

            for i, (token, q) in enumerate(zip(draft_tokens, draft_probs)):
                logits_i = target_logits[n - 1 + i]
                if temperature != 1.0:
                    logits_i = logits_i / temperature
                p = F.softmax(logits_i, dim=-1).to(q.device)

                accepted_token = speculative_sample_one_step(p, q, token)
                accepted.append(accepted_token)

                if accepted_token != token:
                    all_accepted = False
                    break

            # ── PHASE 4: BONUS ──────────────────────────────────────────
            # If all K draft tokens were accepted, sample one free token
            # from the target distribution at position K.
            if all_accepted:
                bonus_logits = target_logits[n - 1 + K]
                if temperature != 1.0:
                    bonus_logits = bonus_logits / temperature
                bonus_p = F.softmax(bonus_logits, dim=-1)
                bonus_token = torch.multinomial(bonus_p, num_samples=1).item()
                accepted.append(bonus_token)

            # Update context and generated list
            generated_ids.extend(accepted)
            new_tokens = torch.tensor([accepted], device=input_ids.device)
            input_ids = torch.cat([input_ids, new_tokens], dim=-1)

            # Reset caches — context has changed, caches are now stale
            draft_past = None
            target_past = None

    return tokenizer.decode(generated_ids[:max_new_tokens], skip_special_tokens=True)


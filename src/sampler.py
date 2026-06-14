

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

def _trim_kv_cache(past_key_values, keep_length):
    """
    Cuts the KV cache down to only the first keep_length positions.

    KV cache is a tuple of (key, value) pairs — one per transformer layer.
    Each key/value tensor has shape [batch, heads, seq_len, head_dim].
    We slice the seq_len dimension to discard rejected-token positions.

    Why: after accepting M of K draft tokens, positions M..K-1 in the cache
    belong to rejected tokens. We must cut them before the next iteration
    so the model doesn't condition on a wrong history.
    """
    return tuple(
        (k[:, :, :keep_length, :], v[:, :, :keep_length, :])
        for k, v in past_key_values
    )


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
    Full speculative decoding loop with persistent KV cache.

    Key difference from the naive version: caches are NEVER reset.
    Instead, after each iteration we trim both caches to exactly the
    accepted prefix. This keeps both models at O(1) cost per iteration
    instead of O(n) — eliminating the O(n²) blowup.

    Phases each iteration:
      1. DRAFT:  run draft model K steps from cached state (feeds 1 token at a time)
      2. TARGET: run target model ONCE on the K draft tokens (not full context)
      3. ACCEPT: rejection sample each draft token, stop at first rejection
      4. BONUS:  if all K accepted, sample one free token from target at position K
      5. UPDATE: trim both caches to accepted prefix; save lookahead logits

    The "lookahead logit" trick: to verify draft_token[0], we need target's
    distribution at the last accepted position. We save this at the end of
    each iteration and carry it into the next one.

    Args:
        draft_model: small fast model (Llama 3.2 1B), on cuda:0
        target_model: large accurate model (Llama 3.1 8B Instruct), on cuda:1
        tokenizer: shared tokenizer
        prompt: input string
        max_new_tokens: total new tokens to generate
        K: number of draft tokens to propose per iteration
        temperature: sampling temperature for both models

    Returns:
        generated text as string
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(draft_model.device)
    generated_ids = []

    with torch.no_grad():

        # ── PRIME BOTH MODELS ON PROMPT ─────────────────────────────────
        # Feed the full prompt once. After this:
        #   draft_past and target_past cover positions 0..prompt_len-1
        #   the logit at the last prompt position = distribution over the
        #   FIRST token we will generate — save these as "lookaheads"
        draft_prime = draft_model(prompt_ids, past_key_values=None, use_cache=True)
        draft_past = draft_prime.past_key_values
        draft_next_logit = draft_prime.logits[0, -1, :]

        target_prime = target_model(
            prompt_ids.to(target_model.device), past_key_values=None, use_cache=True
        )
        target_past = target_prime.past_key_values
        target_lookahead = target_prime.logits[0, -1, :]

        current_len = prompt_ids.shape[1]

        while len(generated_ids) < max_new_tokens:

            # ── PHASE 1: DRAFT ──────────────────────────────────────────
            # Generate K tokens one at a time from the draft model.
            # Each step feeds ONE token and extends the cache by one position.
            # Save all intermediate logits so we can reconstruct any prefix later.
            draft_tokens = []
            draft_qs = []
            draft_logit_at = [draft_next_logit]  # draft_logit_at[i] = dist over token at current_len+i

            for step in range(K):
                logit = draft_logit_at[step]
                if temperature != 1.0:
                    logit = logit / temperature
                q = F.softmax(logit, dim=-1)
                token = torch.multinomial(q, num_samples=1).item()

                draft_tokens.append(token)
                draft_qs.append(q)

                out = draft_model(
                    torch.tensor([[token]], device=draft_model.device),
                    past_key_values=draft_past,
                    use_cache=True
                )
                draft_past = out.past_key_values
                draft_logit_at.append(out.logits[0, -1, :])

            # ── PHASE 2: TARGET ─────────────────────────────────────────
            # Feed ONLY the K draft tokens to the target (not full context).
            # target_past already covers everything before this iteration,
            # so the target only does K new positions of work — O(K), not O(n).
            #
            # Output: target_k_logits[i] = dist over token at current_len+i+1
            # (input token at position current_len+i predicts the token after it)
            draft_tensor = torch.tensor([draft_tokens], device=target_model.device)
            target_out = target_model(
                draft_tensor, past_key_values=target_past, use_cache=True
            )
            target_past = target_out.past_key_values
            target_k_logits = target_out.logits[0]  # shape [K, vocab_size]

            # ── PHASE 3: ACCEPT ─────────────────────────────────────────
            # Verify each draft token via rejection sampling.
            #
            # draft_tokens[0] is verified against target_lookahead — the
            # distribution saved from the previous iteration's last position.
            # draft_tokens[i] (i > 0) is verified against target_k_logits[i-1].
            accepted = []
            M = 0
            all_accepted = True

            tl = target_lookahead / temperature if temperature != 1.0 else target_lookahead
            p0 = F.softmax(tl, dim=-1).to(draft_model.device)
            a0 = speculative_sample_one_step(p0, draft_qs[0], draft_tokens[0])
            accepted.append(a0)
            M = 1

            if a0 != draft_tokens[0]:
                all_accepted = False
            else:
                for i in range(1, K):
                    logit_i = target_k_logits[i - 1]
                    if temperature != 1.0:
                        logit_i = logit_i / temperature
                    p_i = F.softmax(logit_i, dim=-1).to(draft_model.device)
                    a_i = speculative_sample_one_step(p_i, draft_qs[i], draft_tokens[i])
                    accepted.append(a_i)
                    M += 1
                    if a_i != draft_tokens[i]:
                        all_accepted = False
                        break

            # ── PHASE 4: BONUS ──────────────────────────────────────────
            # All K accepted → sample one free token from target at position K.
            # target_k_logits[K-1] = dist over token at current_len+K.
            if all_accepted:
                bonus_logit = target_k_logits[K - 1]
                if temperature != 1.0:
                    bonus_logit = bonus_logit / temperature
                bonus = torch.multinomial(F.softmax(bonus_logit, dim=-1), num_samples=1).item()
                accepted.append(bonus)
                M += 1

            # ── UPDATE STATE ─────────────────────────────────────────────
            generated_ids.extend(accepted)
            current_len += M

            if M <= K:
                # Trim both caches from 0..n+K-1 down to 0..current_len-1.
                # Positions current_len..n+K-1 belong to rejected tokens — discard them.
                draft_past = _trim_kv_cache(draft_past, current_len)
                target_past = _trim_kv_cache(target_past, current_len)

                # Lookaheads for next iteration:
                # draft_logit_at[M] = dist over next token (saved during draft phase)
                # target_k_logits[M-1] = dist over same position from target's view
                draft_next_logit = draft_logit_at[M]
                target_lookahead = target_k_logits[M - 1]

            else:
                # M = K+1: all K accepted plus bonus token b.
                # Caches cover 0..current_len-2 (bonus token b is NOT in either cache yet).
                # Feed b to both models to extend caches and get the next lookaheads.
                b = accepted[-1]

                d_out = draft_model(
                    torch.tensor([[b]], device=draft_model.device),
                    past_key_values=draft_past,
                    use_cache=True
                )
                draft_past = d_out.past_key_values
                draft_next_logit = d_out.logits[0, -1, :]

                t_out = target_model(
                    torch.tensor([[b]], device=target_model.device),
                    past_key_values=target_past,
                    use_cache=True
                )
                target_past = t_out.past_key_values
                target_lookahead = t_out.logits[0, -1, :]

    return tokenizer.decode(generated_ids[:max_new_tokens], skip_special_tokens=True)


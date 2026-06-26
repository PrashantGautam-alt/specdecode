import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


def naive_generate(model, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
    """
    Autoregressive generation written by hand, without calling .generate().

    Runs one forward pass per token. Uses the KV cache so the model only
    processes the newest token each step instead of the full growing sequence.

    Args:
        model: loaded HuggingFace causal LM, already on CUDA in eval mode
        tokenizer: matching tokenizer for the model
        prompt: input string
        max_new_tokens: number of new tokens to generate
        temperature: T less than 1 sharpens the distribution, T greater than 1 flattens it,
                     T approaching 0 becomes greedy

    Returns:
        prompt + generated continuation as a decoded string
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
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

            # feed only the new token next step, the cache holds everything before it
            input_ids = next_token_id.unsqueeze(0)
            generated_ids.append(next_token_id.item())

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def speculative_sample_one_step(p: torch.Tensor, q: torch.Tensor, draft_token: int) -> int:
    """
    One step of speculative decoding rejection sampling.

    Accepts the draft token with probability min(1, p[x] / q[x]).
    On rejection, samples from normalize(max(0, p - q)) instead.
    This guarantees the output distribution equals p exactly, not approximately.

    Args:
        p: target model probability distribution, shape [vocab_size]
        q: draft model probability distribution, shape [vocab_size]
        draft_token: token id proposed by the draft model

    Returns:
        accepted token id as an int
    """
    acceptance_prob = min(1, p[draft_token] / q[draft_token])
    r = torch.rand(1).item()

    if r < acceptance_prob:
        return draft_token

    # draft was overconfident at this token, sample from the residual distribution
    corrected = torch.clamp(p - q, min=0)
    corrected = corrected / corrected.sum()
    return torch.multinomial(corrected, num_samples=1).item()


def _trim_kv_cache(past_key_values, keep_length: int):
    """
    Crops the KV cache to keep_length tokens in place.

    After accepting M tokens in a round, the cache may contain more positions
    than the sequence actually has (leftover draft tokens that were rejected).
    Cropping aligns the cache with the real accepted sequence length.
    """
    past_key_values.crop(keep_length)
    return past_key_values


def speculative_decode(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    K: int = 4,
    temperature: float = 1.0,
    verbose: bool = False,
) -> str:
    """
    Full speculative decoding loop with persistent KV cache.

    Each round:
      1. DRAFT  - run draft model K steps, collect token ids and distributions q
      2. TARGET - run target model once on all K draft tokens, get K distributions p
      3. ACCEPT - rejection sample each draft token, stop at first rejection
      4. UPDATE - trim both caches to accepted length, carry the lookahead logit forward

    The draft cost model: speedup = acceptance_rate / (0.40 * K + 1).
    Each draft pass costs roughly 40 percent of a target pass (single-token overhead
    dominates, not model size). Past K=4 the diminishing acceptance gains stop
    outweighing the guaranteed draft cost.

    Args:
        draft_model: small fast model (Llama 3.2 1B Instruct)
        target_model: large accurate model (Llama 3.1 8B Instruct)
        tokenizer: shared tokenizer (both models use the same vocabulary)
        prompt: input string
        max_new_tokens: total tokens to generate
        K: draft tokens per round
        temperature: applied to both models
        verbose: if True, prints average accepted tokens per round

    Returns:
        generated text as a decoded string
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(draft_model.device)
    generated_ids = []
    current_len = prompt_ids.shape[1]

    with torch.no_grad():
        # prime both caches on the full prompt so the loop never reprocesses it
        draft_prime = draft_model(prompt_ids, past_key_values=None, use_cache=True)
        draft_past = draft_prime.past_key_values
        draft_next_logit = draft_prime.logits[0, -1, :]

        target_prime = target_model(prompt_ids.to(target_model.device), past_key_values=None, use_cache=True)
        target_past = target_prime.past_key_values
        target_lookahead = target_prime.logits[0, -1, :]

        rounds = 0

        while len(generated_ids) < max_new_tokens:
            rounds += 1

            # PHASE 1: DRAFT
            draft_tokens = []
            draft_qs = []
            draft_logit_at = [draft_next_logit]

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
                    use_cache=True,
                )
                draft_past = out.past_key_values
                draft_logit_at.append(out.logits[0, -1, :])

            # PHASE 2: TARGET
            # one forward pass over all K draft tokens gives us K distributions for free
            draft_tensor = torch.tensor([draft_tokens], device=target_model.device)
            target_out = target_model(draft_tensor, past_key_values=target_past, use_cache=True)
            target_past = target_out.past_key_values
            target_k_logits = target_out.logits[0]  # shape [K, vocab_size]

            # PHASE 3: ACCEPT
            accepted = []
            M = 0

            tl = target_lookahead / temperature if temperature != 1.0 else target_lookahead
            p0 = F.softmax(tl, dim=-1).to(draft_model.device)
            a0 = speculative_sample_one_step(p0, draft_qs[0], draft_tokens[0])
            accepted.append(a0)
            M = 1

            if a0 == draft_tokens[0]:
                for i in range(1, K):
                    logit_i = target_k_logits[i - 1]
                    if temperature != 1.0:
                        logit_i = logit_i / temperature
                    p_i = F.softmax(logit_i, dim=-1).to(draft_model.device)
                    a_i = speculative_sample_one_step(p_i, draft_qs[i], draft_tokens[i])
                    accepted.append(a_i)
                    M += 1
                    if a_i != draft_tokens[i]:
                        break

            # UPDATE: trim caches to the real accepted length, carry lookahead forward
            generated_ids.extend(accepted)
            current_len += M
            draft_past = _trim_kv_cache(draft_past, current_len)
            target_past = _trim_kv_cache(target_past, current_len)
            draft_next_logit = draft_logit_at[M]
            target_lookahead = target_k_logits[M - 1]

    if verbose:
        print(f"K={K} avg tokens/round: {len(generated_ids) / rounds:.2f}")

    return tokenizer.decode(generated_ids[:max_new_tokens], skip_special_tokens=True)


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
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    past_key_values = None  # cache starts empty

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

            generated_ids = []
            generated_ids.append(next_token_id.item())

    # 10. Decode the full token sequence back to text.
    #     You need all tokens, not just input_ids (which is now just the last one).
    #     Hint: you need to re-tokenize or keep track of all generated IDs separately.
    #     Think about this one — what do you need to decode the full output?
    text = tokenizer.decode(generated_ids,skip_special_tokens=True)
    return text







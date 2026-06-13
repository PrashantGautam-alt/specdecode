

import torch
from src.sampler import speculative_sample_one_step


def test_rejection_sampling_matches_target_distribution():
    """
    Runs 100K trials of speculative_sample_one_step and checks that the
    empirical token frequencies match the target distribution p within 1%.

    This is the correctness proof — if the math is right, the output
    distribution must equal p exactly regardless of what q is.
    """

    torch.manual_seed(42)

    # Small vocab of 5 tokens to make the test fast and easy to inspect
    p = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05])
    q = torch.tensor([0.1, 0.4, 0.2, 0.2, 0.1])

    num_trials = 100_000
    counts = torch.zeros(5)

    for _ in range(num_trials):
        # Draft model samples one token from q
        draft_token = torch.multinomial(q, num_samples=1).item()

        # Rejection sampling decides what token we actually keep
        accepted_token = speculative_sample_one_step(p, q, draft_token)

        counts[accepted_token] += 1

    empirical = counts / num_trials

    print("\nTarget distribution p:   ", p.tolist())
    print("Empirical distribution:  ", [round(x, 3) for x in empirical.tolist()])
    print("Difference:              ", [round(abs(empirical[i].item() - p[i].item()), 4) for i in range(5)])

    # Check every token is within 1% of target
    for i in range(5):
        diff = abs(empirical[i].item() - p[i].item())
        assert diff < 0.01, f"Token {i}: empirical={empirical[i]:.4f}, target={p[i]:.4f}, diff={diff:.4f}"

    print("\nPASSED — empirical distribution matches p within 1%")


if __name__ == "__main__":
    test_rejection_sampling_matches_target_distribution()



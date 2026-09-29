# Stability: reruns, concurrent load and uncertainty

Follow-up to the benchmark, prompted by a question from Taylor Kolasinski ([Poisson Labs](https://poissonlabs.ai), whose [replay study of Jev](https://poissonlabs.ai/research/the-same-request-twice) found decisions flipping between identical requests): do the local models hold their probabilities across reruns, and how sure are the headline numbers?

All Kev runs are Kev-4B loaded in 4-bit on the same RTX 4060 as the benchmark, reached over the internet through an API gateway, and compared question by question with the original unbatched run in [`results/runs/kev-4b-4bit/jabr-v2`](../results/runs/kev-4b-4bit/jabr-v2).

## Reruns, one request at a time

| Run | Probabilities changed | Top answers flipped | Accuracy |
|---|---|---|---|
| Original (26 Sep 2026) | — | — | 0.8718 |
| rerun-1, rerun-2, rerun-3 (28 Sep 2026) | 0 of 866 | 0 | 0.8718 |

Bit-identical at the four decimal places Kev's API returns. Kev's server caches only the four most recent inputs, so every rerun recomputed every answer.

## Concurrent load (batching)

Kev's server runs requests that arrive together as one GPU batch. [`concurrency.py`](concurrency.py) sends N requests at once; each response's `latency_ms` is the model time of the batch it ran in, which also tells us the batch size.

| Sent at once | Probabilities changed | Largest change | Top answers flipped | Accuracy | Model time per request (p50) | Throughput |
|---|---|---|---|---|---|---|
| 2 | 0 | 0 | 0 | 0.8718 | 187 ms | 5.3/s |
| 4 | 0 | 0 | 0 | 0.8718 | 552 ms | 5.3/s |
| 8 | 314 | 0.018 | 0 | 0.8718 | 1,036 ms | 6.4/s |
| 8, repeated | 321 | 0.018 | 1 | 0.8707 | 1,032 ms | 6.4/s |
| 16 | 588 | 0.023 | 1 | 0.8718 | 1,840 ms | 7.7/s |

- Across all runs, the 2,015 requests that ran alone or in batches of two or three were bit-identical. In batches of five or more, 1,223 of 2,315 changed. (No batch of exactly four formed.)
- The two runs at 8 disagreed with each other on 403 questions: under load, an answer depends on what else shares its batch.
- Decisions barely moved: 2 top-answer flips in 2,598 batched answers, both near-ties. Near a 0.5 threshold, 1 of 279 decisions flipped; near 0.9, 0 of 582.
- Requests were paced to about 55 per minute for the gateway's rate limit, so throughput here measures batching, not peak capacity.

## Uncertainty

From `python stability/analyze.py` (bootstrap over the 866 cases; held-out coverage over 2,000 random half splits):

| Measure | Kev-4B 4-bit | Jev 1.13 |
|---|---|---|
| Accuracy (95% CI) | 0.872 (0.850–0.894) | 0.968 (0.955–0.979) |
| Gap, Jev minus Kev, paired | 9.6 points (7.5–11.8) | |
| Confidently wrong (≥0.9 and wrong) | 1 of 866 (1 of 200 confident answers) | 2 of 866 (2 of 652) |
| Automatable at 5% error, in-sample | 75.4% | 100% |
| Same, threshold chosen on the other half | 75.5% (68.4–80.8%) | 100% (99.1–100%) |
| Realised error on those | 4.9% (2.0–8.7%) | 3.0% (1.9–4.2%) |

“Automatable at 5% error” doesn't use a fixed 0.9 threshold: it accepts the most confident answers while errors among them stay within 5%.

## Julia-1 checks

[Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) (Supersonic Labs, revision `a85b1273`, weights SHA-256 `df853bf7…`) was scored through [`adapters/julia_server.py`](../adapters/julia_server.py) on CPU. Its main results are in [`results/runs/julia-1`](../results/runs/julia-1). Here:

- `rerun-2`, `rerun-3`: identical to the first run (deterministic).
- `noul-criteria`: the same suite with `{"false": "No", "true": "Yes"}` added to every yes/no question ([`add_noul_criteria.py`](add_noul_criteria.py)). Accuracy stayed at 0.449, but 101 answers flipped: its yes/no answers depend heavily on wording.

## Reproduce

```bash
python stability/rerun.py --endpoint <url> --model kev-latest --suite suites/jabr-v2.jsonl --out <dir> [--api-key-file F] [--rate 55]
python stability/concurrency.py --endpoint <url> --suite suites/jabr-v2.jsonl --out <dir> --concurrency 8 [--api-key-file F]
python stability/analyze.py
```

`--user-agent` is available on both clients for gateways that block script user agents.

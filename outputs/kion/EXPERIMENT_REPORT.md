# KION — Java GPT-2-style + reward model + policy-gradient RL

## Data and protocol

- Dataset: official MTS KION interactions.csv
- Dates: train before 2021-08-09; validation until 2021-08-16; test thereafter.
- Selected users: 1800; train-only candidate catalog: 160.
- Examples: train 650, validation 120, test 87.
- Excluded held-out items not in train catalog (including train samples): 7493.
- Histories include events from **strictly earlier calendar days**; no within-day order is assumed.
- One target per user in each held-out split; each policy uses the same evaluation examples.

## Reward model

Observed validation: MAE=0.38760, RMSE=0.41461, N=120.  
Observed test: MAE=0.36391, RMSE=0.39263, N=87.

The reward model is a history-conditioned latent-factor sigmoid regressor predicting capped watched_pct/100. Its predictions for novel policy choices are **not observed counterfactual user rewards**.

## Comparison (same held-out test, no test-driven selection)

| Policy | Recall@10 | NDCG@10 | Proxy reward | Coverage@10 | Top1 max share | Repeat@10 |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 0.2184 | 0.1332 | 0.5109 | 0.2125 | 0.8736 | 0.0701 |
| Reward reranking | 0.1379 | 0.0839 | 0.6112 | 0.1625 | 0.9770 | 0.0368 |
| RL-lowKL | 0.2069 | 0.1213 | 0.5399 | 0.2375 | 0.5287 | 0.0655 |
| RL-highKL | 0.2184 | 0.1330 | 0.5109 | 0.2188 | 0.8736 | 0.0655 |
| RL-novelty | 0.2184 | 0.1295 | 0.5249 | 0.2313 | 0.6207 | 0.0667 |

## Five-step autoregressive rollout: mean *predicted* reward

| Policy | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---:|---:|---:|---:|---:|
| SFT | 0.5109 | 0.5131 | 0.5136 | 0.5151 | 0.5167 | 
| RL-lowKL | 0.5399 | 0.5525 | 0.5533 | 0.5553 | 0.5560 | 
| RL-highKL | 0.5109 | 0.5131 | 0.5136 | 0.5151 | 0.5167 | 
| RL-novelty | 0.5249 | 0.5318 | 0.5279 | 0.5338 | 0.5351 | 

**Interpretation:** A generated recommendation is NOT a confirmed play. Appending it to a history creates counterfactual states; compounding errors and reward-model extrapolation make these rollouts only a diagnostic.

## What this experiment can and cannot conclude

The REINFORCE runs update the actual Transformer parameters using a sampled policy gradient, an exact categorical KL term relative to frozen SFT, and a supervised CE anchor. The reward model is frozen during RL. Any increase in its predicted score is **not** proof of improved watch time or retention. Observational logs lack randomized action propensities. Online A/B testing is required for causal conclusions. Long-term objectives might use next-session return and total subsequent meaningful interactions, but these are not established by this offline experiment.

## Limitations

- This is a tiny **GPT-2-style** decoder (one causal pre-LN Transformer block), not official pretrained GPT-2.
- CPU-only minimal Java autodiff prioritizes auditability and limited execution time; the sampled catalog makes metrics incomparable to full benchmark results.
- Test is warm-catalog conditional: unseen items are excluded. Same-day interactions are unordered.
- KL regularization is on the sampled history states, not a guarantee of unchanged production policy.
- Training is stochastic; seed and run settings should be recorded. Repeat with multiple seeds for stronger conclusions.

# KION — Java GPT-2-style + reward model + policy-gradient RL

## Data and protocol

- Dataset: official MTS KION interactions.csv
- Dates: train before 2021-08-09; validation until 2021-08-16; test thereafter.
- Selected users: 5000; train-only candidate catalog: 300.
- Examples: train 2000, validation 350, test 314.
- Excluded held-out items not in train catalog (including train samples): 17067.
- Histories include events from **strictly earlier calendar days**; no within-day order is assumed.
- One target per user in each held-out split; each policy uses the same evaluation examples.

## Reward model

Observed validation: MAE=0.35946, RMSE=0.39212, N=350.  
Observed test: MAE=0.36581, RMSE=0.39391, N=314.

The reward model is a history-conditioned latent-factor sigmoid regressor predicting capped watched_pct/100. Its predictions for novel policy choices are **not observed counterfactual user rewards**.

## Comparison (same held-out test, no test-driven selection)

| Policy | Recall@10 | NDCG@10 | Proxy reward | Coverage@10 | Top1 max share | Repeat@10 |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 0.2261 | 0.1140 | 0.5887 | 0.0600 | 0.3535 | 0.0768 |
| Reward reranking | 0.1656 | 0.1064 | 0.6429 | 0.0433 | 1.0000 | 0.0640 |
| RL-lowKL | 0.1815 | 0.0962 | 0.5989 | 0.1300 | 0.4459 | 0.0691 |
| RL-highKL | 0.2038 | 0.1122 | 0.6073 | 0.0767 | 0.4682 | 0.0752 |
| RL-novelty | 0.2134 | 0.1165 | 0.6119 | 0.0767 | 0.5350 | 0.0748 |

## Five-step autoregressive rollout: mean *predicted* reward

| Policy | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---:|---:|---:|---:|---:|
| SFT | 0.5887 | 0.5716 | 0.5946 | 0.5966 | 0.5980 | 
| RL-lowKL | 0.5989 | 0.5896 | 0.5957 | 0.5974 | 0.5986 | 
| RL-highKL | 0.6073 | 0.5776 | 0.5954 | 0.5972 | 0.5985 | 
| RL-novelty | 0.6119 | 0.5911 | 0.5962 | 0.5978 | 0.5990 | 

**Interpretation:** A generated recommendation is NOT a confirmed play. Appending it to a history creates counterfactual states; compounding errors and reward-model extrapolation make these rollouts only a diagnostic.

## What this experiment can and cannot conclude

The REINFORCE runs update the actual Transformer parameters using a sampled policy gradient, an exact categorical KL term relative to frozen SFT, and a supervised CE anchor. The reward model is frozen during RL. Any increase in its predicted score is **not** proof of improved watch time or retention. Observational logs lack randomized action propensities. Online A/B testing is required for causal conclusions. Long-term objectives might use next-session return and total subsequent meaningful interactions, but these are not established by this offline experiment.

## Limitations

- This is a tiny **GPT-2-style** decoder (one causal pre-LN Transformer block), not official pretrained GPT-2.
- CPU-only minimal Java autodiff prioritizes auditability and limited execution time; the sampled catalog makes metrics incomparable to full benchmark results.
- Test is warm-catalog conditional: unseen items are excluded. Same-day interactions are unordered.
- KL regularization is on the sampled history states, not a guarantee of unchanged production policy.
- Training is stochastic; seed and run settings should be recorded. Repeat with multiple seeds for stronger conclusions.

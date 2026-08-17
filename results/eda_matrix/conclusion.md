## What we found

- **Stage 1 champion:** property graph (PR-AUC 0.208 on test month 7), compared with tabular anchor PR-AUC 0.211.
- Highest PR-AUC this run was **none** (0.211); property graph is still the locked champion.
- **Property graph** recall was 27.2% vs anchor 29.1%.
- **Temporal kNN** did not beat the tabular baseline on PR-AUC in this run.
- **Preprocessing ablation:** best combo was Yeo-Johnson off, SMOTE off (PR-AUC 0.213).
- **Variant benchmark:** PR-AUC ranged from 0.059 to 0.514 across 6 BAF variants (std 0.160).
- **Cost model (anchor scores):** optimal cutoffs block at 0.85 and alert at 0.83. Expected profit is about HKD 38.48 on test month 7 — a rough proxy, not real bank savings.

## What to do next

- Use **property graph** features as the default for the next modeling stage.
- Replace the fraud revenue proxy with real loss data before trusting the cost model.
- Re-run ablations if you change preprocessing or download new BAF variants.
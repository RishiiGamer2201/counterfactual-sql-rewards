# Counterfactual-database rewards for text-to-SQL RL

Code, results and paper for **Single-Database Rewards Are Hackable: Counterfactual Execution for Trustworthy Text-to-SQL RL** (Amazon Trusted AI Symposium 2027 submission). Paper: [paper/main.pdf](paper/main.pdf). Adapters, data and demo: [hf.co/counterfactual-sql](https://huggingface.co/counterfactual-sql).

> RL for text-to-SQL agents usually pays a query when its result matches the reference on one database, but a wrong query can match by luck when the data lack the rows that expose its bug. We study a counterfactual-database (CF) reward that also runs the final query on a few hidden, code-generated copies of the database, some changing the reference answer and some preserving it, and pays only if all results match (strict form) or by the share of copies matched (mean form). On 1,302 synthetic audit questions, the strict form lowers the share of buggy queries paid in full from 29.5% to 16.8% and of reward hacks from 54.1% to 1.9%, rejecting 0 of 2,847 correct queries; copies from a generator blind to our bug families do as well. In a controlled GRPO comparison on a 3.35B model with three seeds per arm, training on the mean form did not improve in-domain accuracy (8.4% vs 11.7%), because the two rewards disagreed on only 5–7% of rollouts; on Spider it gained 1.6 points. One seed reverses both signs, so single-seed reward comparisons mislead.

## Main results

| Result | Number (95% CI) |
|---|---|
| Buggy queries (mutants) paid in full: single-DB vs CF reward (blind copies) | 29.5% vs 16.8% |
| Reward hacks (literal smuggling, hard-coded IDs, ...) paid: single-DB vs CF | 54.1% vs 1.9% |
| Correct queries rejected by the CF reward | 0 of 2,847 |
| AgentTune's built-in SQL reward on `WHERE 1 = 1` + always "yes" | same reward as honest SQL; fixed upstream with our patch |
| Set-compare reward on one DB (BIRD / SkyRL-SQL style): mutants / hacks paid | 37.5% / 54.1% |
| Spider dev, 8 published systems: EX passes that fail a test-suite DB | 8.1–14.9% |
| GRPO on CF (mean) vs single-DB reward, verified test accuracy, 3 seeds pooled | 8.4% vs 11.7% (−3.2 [−4.7, −1.8]) |
| Same pair, Spider dev test-suite accuracy, 3 seeds pooled | 29.3% vs 27.8% (+1.6 [+0.8, +2.3]) |
| Rollouts on which the two rewards disagreed during training | about 5–7% |
| Planted stale-cache pilot (one seed): cache reads on test, SFT / single-DB / CF | 16.9% / 2.1% / 1.5% |

The paper's tables are generated from `results/`, and `paper/scripts/check_numbers.py` checks every number in the text against them.

## Setup (Linux / WSL)

```bash
# AgentTune trainer (not committed), pinned and patched
mkdir -p vendor && git clone https://github.com/Lexsi-Labs/AgentTune.git vendor/AgentTune
git -C vendor/AgentTune checkout d4db4c1 && git -C vendor/AgentTune apply ../../patches/agenttune-multiturn-ids.patch
# CPU env (data, rewards, audit, scoring)
uv venv .venv && uv pip install --python .venv/bin/python pandas numpy scipy matplotlib pytest
# training env (GPU): python>=3.12, AgentTune + TRL + vLLM 0.22.1
uv venv --python 3.12 .venv-train && uv pip install --python .venv-train/bin/python -e vendor/AgentTune trl peft datasets vllm==0.22.1
```

Tiny Aya Global (`CohereLabs/tiny-aya-global`) is gated: accept its licence and use a Hugging Face read token.

## Rebuild data (CPU, deterministic)

```bash
.venv/bin/python -m sqlcf.cf --name audit --split all   --seeds 301 302 303 304 305 --size S
.venv/bin/python -m sqlcf.cf --name train --split train --seeds 101 102 103 104 105 106 --size S
.venv/bin/python -m sqlcf.cf --name test  --split test  --seeds 501 502 503 --size S --test
.venv/bin/python -m sqlcf.build_sft --items data/train/items.jsonl --out data/sft.jsonl
.venv/bin/python -m pytest -q tests
```

## Experiments

```bash
.venv/bin/python -m sqlcf.audit e0                                         # reward audit -> results/
.venv/bin/python -m sqlcf.audit agenttune                                  # AgentTune's reward
.venv/bin/python -m sqlcf.audit e1 --seeds 900 901 902                     # blind rate vs database size
.venv-train/bin/python train.py sft  --out runs/sft                        # SFT
.venv-train/bin/python train.py grpo --reward R_exec_rex --out runs/grpo_B # GRPO, single-DB reward
.venv-train/bin/python train.py grpo --reward R_cf_mean  --out runs/grpo_C # GRPO, CF reward
for m in sft grpo_B grpo_C; do .venv-train/bin/python evaluate.py gen --model runs/$m/merged --name ${m#grpo_}; done
.venv/bin/python evaluate.py score --traj results/traj_*.jsonl             # test-set metrics
.venv/bin/python scripts/seeds_eval.py                                     # seeds 0-2, paired bootstrap
bash scripts/local_spider.sh                                               # Spider dev transfer (generation + scoring)
.venv/bin/python scripts/a1_mechanism.py                                   # why training differs little
.venv/bin/python scripts/a4_baselines.py                                   # family-blind copies, held-out family, larger DB
.venv/bin/python scripts/audit_skyrl.py                                    # SkyRL-SQL reward audit
.venv/bin/python scripts/agenttune_repro.py                                # AgentTune reward exploit, before / after patch
.venv/bin/python scripts/a2_spider_lucky.py                                # published Spider systems (needs test-suite DBs + ESM+ predictions)
.venv/bin/python scripts/audit_sample.py && .venv/bin/python scripts/audit_labels.py  # hand audit
python3 scripts/leak_check.py                                              # train/test/audit overlap
.venv/bin/python verify_all.py                                             # re-derive results/*.json (needs runs/ from the steps above)
cd paper && bash build.sh                                                  # tables, PDF, number check
```

Stale-cache pilot: `scripts/leaky_build.py` (data), `scripts/leaky_sft.py` / `scripts/local_leaky_sft.sh` (hack-seeded SFT), then GRPO as above with `SQLCF_DATA=data_leaky`, and `scripts/leaky_eval.py` -> `results/leaky_eval.json`.

Spider evaluation needs the test-suite databases of Zhong et al. (2020) in `external/testsuite_db/database`, and the published-system audit needs [ESM+](https://github.com/bossben/ESMp) (commit 91dd85b) in `external/ESMp`.

## Layout

`sqlcf/` data generator, templates, counterfactual databases, comparator, rewards, audit, agent tool · `train.py` SFT, GRPO, merge · `evaluate.py` generation and scoring · `scripts/` experiments · `results/` stored outputs · `patches/` AgentTune fixes · `demo/` reward demo page · `paper/` LaTeX source · `tests/` reward tests.

## Licence

Code: [Apache-2.0](LICENSE). Adapters, data and rollouts: CC-BY-NC-4.0 ([text](docs/LICENSE-CC-BY-NC-4.0.txt)), from Tiny Aya Global's licence; Spider trajectories are CC BY-SA 4.0.

"""Stage and push the Hugging Face release to the org `counterfactual-sql`: three LoRA adapters (SFT, GRPO with the
single-DB reward, GRPO with the counterfactual reward), the data/rollouts dataset, and the in-browser demo Space.
Needs local data/ and runs/ (README steps). Everything is created private; --public flips visibility.

  HF_TOKEN=... python scripts/hf_release.py [--stage-only] [--public] [--only models dataset space]
"""
import argparse
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORG = "counterfactual-sql"
STAGE = ROOT / "runs/hf_stage"
GITHUB = "https://github.com/RishiiGamer2201/counterfactual-sql-rewards"
DATA, SPACE = f"{ORG}/counterfactual-sql-data", f"{ORG}/counterfactual-sql-demo"
MODELS = {  # repo: (run dir, e4_eval key, role)
    "tiny-aya-global-sql-sft": ("runs/sft", "sft_ho", "SFT warm start on scripted trajectories (3 training templates held out for GRPO); the initialisation of both GRPO adapters"),
    "tiny-aya-global-sql-grpo-single-db": ("runs/grpo_B", "B", "GRPO, 60 steps, single-database strict execution reward (`R_exec_rex`), seed 0"),
    "tiny-aya-global-sql-grpo-cf": ("runs/grpo_C", "C", "GRPO, 60 steps, counterfactual-database reward, mean form (`R_cf_mean`), seed 0"),
}
LINKS = f"""Paper: *Single-Database Rewards Are Hackable: Counterfactual Execution for Trustworthy Text-to-SQL RL*
(Shreshth Rai, Rishii Kumar Singh, Aksh Aggarwal; AIMS-DTU, Delhi Technological University).
[Code and paper]({GITHUB}) · [Data](https://huggingface.co/datasets/{DATA}) · [Demo](https://huggingface.co/spaces/{SPACE}) ·
Adapters: {" · ".join(f"[{r.removeprefix('tiny-aya-global-sql-')}](https://huggingface.co/{ORG}/{r})" for r in MODELS)}
"""
NOTICE = """This adapter is a derivative of CohereLabs/tiny-aya-global and is released under the same terms:
CC-BY-NC-4.0 (full text in LICENSE) with Cohere's acceptable-use addendum, https://cohere.com/c4ai-cc-by-nc-license.
Non-commercial use only. You also need access to the gated base model.
"""
BANNED = re.compile(os.environ.get("RELEASE_BANNED", r"/home/"), re.I)  # extra private words via env


def model_card(repo, key, role):
    v = json.loads((ROOT / "results/e4_eval.json").read_text())[key]
    pct = lambda x: f"{100 * x[0]:.1f}% [{100 * x[1]:.1f}, {100 * x[2]:.1f}]"
    return f"""---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
license_link: https://cohere.com/c4ai-cc-by-nc-license
tags: [text-to-sql, grpo, lora, reward-hacking, counterfactual-rewards]
datasets: [{DATA}]
---
# {repo}

LoRA adapter (r = 16) for [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global): {role}.

{LINKS}
Test set: 331 questions x 8 samples, T = 0.7, fresh databases from a different generator. Verified = correct on the
original database and all 4 hidden test copies.

| Metric | Value (95% Wilson CI) |
|---|---|
| Verified accuracy | {pct(v['verified'])} |
| Single-DB accuracy | {pct(v['single'])} |
| Silent-error rate | {pct(v['silent_error'])} |

Seeds 1-2 and the pooled comparison are in the paper and in `results/seeds_eval.json`.

## Use

Accept the [base model's licence](https://huggingface.co/CohereLabs/tiny-aya-global), then:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{ORG}/{repo}")
tok = AutoTokenizer.from_pretrained("{ORG}/{repo}")
```

It is a multi-turn SQL agent with one `run_sql` tool (prompt and parser: `sqlcf/agent.py`, `evaluate.py gen`).
For vLLM, merge first (`python train.py merge --init <adapter> --out merged`), since vLLM LoRA fails on the tied embeddings.

## Licence
{NOTICE}"""


DATA_CARD = f"""---
license: cc-by-nc-4.0
language: [en]
tags: [text-to-sql, reward-modeling, reward-hacking, grpo, counterfactual, sqlite]
pretty_name: Counterfactual-database rewards for text-to-SQL RL
---
# Counterfactual-database rewards: data, results and rollouts

{LINKS}
Every number in the paper is re-derived from these files (`verify_all.py` and `paper/scripts/check_numbers.py` in the code repo).

```python
import tarfile
from huggingface_hub import snapshot_download

d = snapshot_download("{DATA}", repo_type="dataset")
for split in ("audit", "train", "test"):
    tarfile.open(f"{{d}}/{{split}}/dbs.tar.gz").extractall(f"{{d}}/{{split}}")  # SQLite DBs next to items.jsonl
```

| Path | What |
|---|---|
| `{{audit,train,test}}/items.jsonl` | Questions, reference SQL, database paths, reference results on the original and the copies |
| `{{audit,train,test}}/dbs.tar.gz` | SQLite databases: originals and counterfactual copies |
| `sft*.jsonl`, `dpo_*.jsonl` | SFT trajectories and DPO pairs |
| `results/` | Every result file of the paper: reward audits, test trajectories (`traj_<model>.jsonl`), Spider runs, seeds, LeakySQL pilot |
| `runs/grpo_<arm>_rewards.jsonl` | Every GRPO rollout with its final SQL, reward and per-database pass vector (seeds 0-2, curriculum, LeakySQL) |
| `runs/traj_sft_ho_grpoprompts.jsonl` | SFT rollouts on the GRPO prompts, replayed by the partial-credit and period-copy analyses |

Synthetic retail data from a seeded generator; no personal data. Rollouts contain outputs of Tiny Aya Global
(CC-BY-NC-4.0) and Cohere Command A, hence the non-commercial licence. `results/spider_*.jsonl` hold rows from Spider
databases (Yu et al. 2018; test-suite DBs from Zhong et al. 2020) and are CC BY-SA 4.0.
"""

SPACE_CARD = f"""---
title: Counterfactual SQL Rewards
emoji: 🧮
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
license: cc-by-nc-4.0
short_description: Check SQL on hidden database copies the agent never saw
tags: [text-to-sql, reward-hacking]
---
A SQL query can match the answer on the one database an agent explored and still be wrong. Pick a test question and
a recorded run (base, SFT, GRPO with the single-DB reward, GRPO with the counterfactual reward, Command A), or type your
own SQL, and see it checked on the original database plus hidden copies, with the score each reward gives.
Runs in your browser (Pyodide); the first load takes 20-60 s.

{LINKS}"""


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def stage_models():
    for repo, (run, key, role) in MODELS.items():
        d, src = STAGE / "models" / repo, ROOT / run / "adapter"
        for f in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jsonl", "chat_template.jinja", "special_tokens_map.json"):
            if (src / f).exists():
                cp(src / f, d / f)
        for f in ("grpo_training_config.yaml", "training_stats.json"):
            if (ROOT / run / f).exists():
                cp(ROOT / run / f, d / f)
        cp(ROOT / "docs/LICENSE-CC-BY-NC-4.0.txt", d / "LICENSE")
        (d / "NOTICE").write_text(NOTICE)
        (d / "README.md").write_text(model_card(repo, key, role))
        assert json.loads((d / "adapter_config.json").read_text())["base_model_name_or_path"] == "CohereLabs/tiny-aya-global"


def stage_dataset():
    d = STAGE / "dataset"
    for split in ("audit", "train", "test"):
        for f in (ROOT / "data" / split).glob("*.json*"):
            cp(f, d / split / f.name)
        with tarfile.open(d / split / "dbs.tar.gz", "w:gz") as t:
            t.add(ROOT / "data" / split / "dbs", arcname="dbs")
    for f in ("sft.jsonl", "sft_all_templates.jsonl", "sft_distil.jsonl", "dpo_std.jsonl", "dpo_cf.jsonl"):
        cp(ROOT / "data" / f, d / f)
    for f in (ROOT / "results").rglob("*"):
        if f.is_file() and not f.name.startswith("_"):
            cp(f, d / "results" / f.relative_to(ROOT / "results"))
    for f in [*(ROOT / "runs").glob("grpo_*_rewards.jsonl"), ROOT / "runs/traj_sft_ho_grpoprompts.jsonl"]:
        cp(f, d / "runs" / f.name)
    (d / "README.md").write_text(DATA_CARD)


def stage_space():
    d = STAGE / "space"
    files = ["sqlcf/demo.py", "sqlcf/rewards.py", "sqlcf/compare.py", "data/test/items.jsonl",
             *[f"results/traj_{m}.jsonl" for m in ("base", "sft_ho", "B", "C", "command_a")],
             *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data/test/dbs").rglob("*.db"))]
    for rel in files:
        cp(ROOT / rel, d / rel)
    cp(ROOT / "demo/index.html", d / "index.html")
    (d / "files.json").write_text(json.dumps(files))
    (d / "README.md").write_text(SPACE_CARD)


def scan(d):  # refuse to publish text matching BANNED (local paths and any words passed in RELEASE_BANNED)
    bad = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix in {".md", ".json", ".jsonl", ".yaml", ".html", ".py", ".txt", ""} \
                and p.name != "tokenizer.json":  # base model's vocabulary, not our text
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                if BANNED.search(line):
                    bad.append(f"{p.relative_to(STAGE)}:{i}: {line[:120]}")
    return bad


ap = argparse.ArgumentParser()
ap.add_argument("--stage-only", action="store_true")
ap.add_argument("--public", action="store_true")
ap.add_argument("--only", nargs="+", default=["models", "dataset", "space"], choices=["models", "dataset", "space"])
a = ap.parse_args()

shutil.rmtree(STAGE, ignore_errors=True)
for part in a.only:
    {"models": stage_models, "dataset": stage_dataset, "space": stage_space}[part]()
bad = scan(STAGE)
for b in bad[:20]:
    print("BANNED", b)
files = [p for p in STAGE.rglob("*") if p.is_file()]
print(f"staged {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB in {STAGE}; banned hits: {len(bad)}")
if bad:
    sys.exit(1)
if a.stage_only:
    sys.exit(0)

from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=os.environ.get("HF_TOKEN") or sys.exit("set HF_TOKEN"))
targets = ([(f"{ORG}/{r}", "model", STAGE / "models" / r) for r in MODELS] if "models" in a.only else []) \
    + ([(DATA, "dataset", STAGE / "dataset")] if "dataset" in a.only else []) \
    + ([(SPACE, "space", STAGE / "space")] if "space" in a.only else [])
for rid, kind, folder in targets:
    api.create_repo(rid, repo_type=kind, private=not a.public, exist_ok=True, **({"space_sdk": "static"} if kind == "space" else {}))
    api.upload_folder(repo_id=rid, repo_type=kind, folder_path=str(folder), commit_message="release")
    if a.public:
        api.update_repo_settings(rid, repo_type=kind, private=False, **({"gated": "auto"} if kind == "model" else {}))
    print(f"https://huggingface.co/{'' if kind == 'model' else kind + 's/'}{rid}")

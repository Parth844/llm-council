"""GSM8K eval: single best model vs full council.

Usage:
    uv run python eval/run_eval.py --n 100 --rounds 2

Sequential over questions (rate-limit friendly); all responses cached to
eval/cache/ so re-runs are free. Writes a markdown report to eval/results.md.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from council import prompts  # noqa: E402
from council.caching import CachingClient  # noqa: E402
from council.client import ModelUnavailableError, NIMClient  # noqa: E402
from council.config import load_api_key, load_config  # noqa: E402
from council.engine import DebateEngine  # noqa: E402
from council.extract import extract_number  # noqa: E402

CACHE_DIR = ROOT / "eval" / "cache"
RESULTS_PATH = ROOT / "eval" / "results.md"
GOLD_RE = re.compile(r"####\s*(-?[\d,\.]+)")


def load_gsm8k(n: int) -> list[dict[str, float | str]]:
    from datasets import load_dataset  # heavy import, keep local

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds.select(range(n)):
        m = GOLD_RE.search(row["answer"])
        if not m:
            continue
        items.append(
            {"question": row["question"], "gold": float(m.group(1).replace(",", ""))}
        )
    return items


def correct(text: str | None, gold: float) -> bool:
    if text is None:
        return False
    num = extract_number(text)
    return num is not None and abs(num - gold) < 1e-6


async def main() -> int:
    parser = argparse.ArgumentParser(description="GSM8K: single model vs council")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--config", default=str(ROOT / "config" / "models.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    client = CachingClient(
        NIMClient(
            api_key=load_api_key(),
            base_url=config.base_url,
            requests_per_minute=config.requests_per_minute,
        ),
        CACHE_DIR,
    )
    single_model = config.council[0]  # "best single model" = first council entry
    print(f"Loading GSM8K (first {args.n} of test split)…")
    items = load_gsm8k(args.n)
    print(f"{len(items)} questions | single model: {single_model.id} | "
          f"council: {len(config.council)} models, {args.rounds} rounds")

    single_ok = 0
    single_latency = 0.0
    council_ok = 0
    council_latency = 0.0
    flips_wrong_to_right = 0
    flips_right_to_wrong = 0

    try:
        for i, item in enumerate(items):
            q, gold = str(item["question"]), float(item["gold"])

            # --- (a) single model ---
            t0 = time.monotonic()
            try:
                res = await client.chat(
                    single_model.id,
                    [{"role": "system", "content": prompts.ROUND1_SYSTEM},
                     {"role": "user", "content": q}],
                    temperature=single_model.temperature,
                    max_tokens=single_model.max_tokens,
                )
                single_ok += correct(res.text, gold)
            except ModelUnavailableError as exc:
                print(f"  [q{i}] single model failed: {exc}")
            single_latency += time.monotonic() - t0

            # --- (b) full council ---
            t0 = time.monotonic()
            engine = DebateEngine(config, client, skip_health_check=True)
            result = await engine.run(q, rounds=args.rounds, session_id=f"eval-{i}")
            council_latency += time.monotonic() - t0
            council_ok += correct(result.synthesis, gold)

            # flip analysis: each model's round-1 answer vs its final answer
            round1 = {e.alias: e.text for e in result.events
                      if e.type == "model_answered" and e.round == 1}
            for alias, final_text in result.answers.items():
                if alias not in round1:
                    continue
                before = correct(round1[alias], gold)
                after = correct(final_text, gold)
                flips_wrong_to_right += (not before) and after
                flips_right_to_wrong += before and (not after)

            done = i + 1
            print(f"  [{done}/{len(items)}] single {single_ok}/{done} | "
                  f"council {council_ok}/{done} | cache {client.hits}H/{client.misses}M")
    finally:
        await client.aclose()

    n = len(items)
    report = f"""# LLM Council — GSM8K Eval Results

- Dataset: GSM8K test split, first {n} questions
- Single model: `{single_model.id}`
- Council: {len(config.council)} models × {args.rounds} rounds + Chief Justice
- API calls this run: {client.misses} (cache hits: {client.hits})

| Setup | Accuracy | Avg latency / question |
|---|---|---|
| Single best model | {single_ok}/{n} ({100 * single_ok / n:.1f}%) | {single_latency / n:.1f}s |
| Full council | {council_ok}/{n} ({100 * council_ok / n:.1f}%) | {council_latency / n:.1f}s |

## Flip analysis (cross-examination effect, per model per question)

| Direction | Count |
|---|---|
| wrong → right | {flips_wrong_to_right} |
| right → wrong | {flips_right_to_wrong} |
| Net gain | {flips_wrong_to_right - flips_right_to_wrong} |
"""
    RESULTS_PATH.write_text(report)
    print(f"\nWrote {RESULTS_PATH}")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

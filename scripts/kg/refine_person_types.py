"""Post-build pass: demote generic "person" entities to OTHER.

LLM extraction tags person-CATEGORIES (Barn, Forskare, Kvinnor, Politiker) and
even pronouns (Jag) as PERSON alongside real named individuals — a known
weakness, amplified in first-person Swedish speech. Rather than re-extracting
the corpus, this sends only the distinct PERSON *names* to gemma in batches
and asks which are NOT specific named individuals; the result is an overrides
file the adapter applies on a re-fold (``adapter.py --type-overrides``).

Runs in the PROJECT venv (lance + httpx only):

    uv run python scripts/kg/refine_person_types.py \
        --db transcripts_v2.lance --out kg_work/person_overrides.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx
import lance

PROMPT = """Du får en lista med entitetsnamn från svenska presskonferens-transkript.
Vilka av namnen är INTE en specifik namngiven person? (t.ex. yrken/roller som
"Forskare", grupper som "Barn" eller "Kvinnor", pronomen som "Jag", kategorier
som "Högutbildade"). Riktiga personnamn som "Anna Lindberg" eller enstaka
förnamn som "Ingegerd" ska INTE tas med.

Svara ENBART med en JSON-lista av namnen som inte är specifika personer.

Namn:
{names}"""


def classify_batch(client: httpx.Client, url: str, model: str, names: list[str]) -> set[str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(names="\n".join(f"- {n}" for n in names))}],
        "max_tokens": 2000,
        "temperature": 0.0,
    }
    resp = client.post(f"{url}/chat/completions", json=body, timeout=120)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return set()
    generic = json.loads(match.group(0))
    # only accept names that were actually in the batch (no hallucinated extras)
    allowed = {n.strip().lower() for n in names}
    return {n for n in generic if isinstance(n, str) and n.strip().lower() in allowed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Demote generic PERSON entities to OTHER.")
    parser.add_argument("--db", default="transcripts_v2.lance")
    parser.add_argument("--out", default="kg_work/person_overrides.json")
    parser.add_argument("--gemma-url", default="http://localhost:8003/v1")
    parser.add_argument("--gemma-model", default="google/gemma-4-31B-it")
    parser.add_argument("--batch", type=int, default=80)
    args = parser.parse_args()

    tbl = lance.dataset(str(Path(args.db) / "kg_entities.lance")).to_table().to_pylist()
    persons = [(r["entity_id"], r["name"]) for r in tbl if r["entity_type"] == "PERSON"]
    print(f"{len(persons)} PERSON entities to classify")

    overrides: dict[str, str] = {}
    with httpx.Client() as client:
        for start in range(0, len(persons), args.batch):
            batch = persons[start : start + args.batch]
            names = [n for _, n in batch]
            try:
                generic = classify_batch(client, args.gemma_url, args.gemma_model, names)
            except Exception as exc:  # one retry, then skip the batch (stays PERSON)
                print(f"batch {start}: retrying after {exc}")
                try:
                    generic = classify_batch(client, args.gemma_url, args.gemma_model, names)
                except Exception as exc2:
                    print(f"batch {start}: SKIPPED ({exc2})")
                    continue
            lowered = {g.strip().lower() for g in generic}
            for eid, name in batch:
                if name.strip().lower() in lowered:
                    overrides[eid] = "OTHER"
            print(f"batch {start + len(batch)}/{len(persons)}: {len(overrides)} demoted so far", flush=True)

    Path(args.out).write_text(json.dumps(overrides, indent=0))
    print(f"wrote {len(overrides)} overrides -> {args.out}")
    demoted = [n for eid, n in persons if eid in overrides]
    print("examples:", demoted[:12])


if __name__ == "__main__":
    main()

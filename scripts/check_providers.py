"""Live check of configured LLM providers/models (no secrets printed).

  python scripts/check_providers.py            # availability only
  python scripts/check_providers.py --ping     # also send a 1-token request to each available benchmark model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402,F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true")
    a = ap.parse_args()
    from common.config import get_model_config
    from llm.router import get_router

    r = get_router()
    out = {"routing": r.status(), "benchmark": []}
    for b in get_model_config().get("benchmark", []):
        prov = r.providers.get(b["provider"])
        row = {**b, "available": bool(prov and prov.available(b["model"]))}
        if a.ping and row["available"]:
            try:
                res = r.complete("explainer", [{"role": "user", "content": "Reply with OK."}], provider=b["provider"],
                                 model=b["model"], max_tokens=5, experiment="provider_check")
                row.update(ok=True, latency_ms=round(res.latency_ms, 1))
            except Exception as e:
                row.update(ok=False, error=str(e)[:200])
        out["benchmark"].append(row)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

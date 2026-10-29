# LLMs, prompts, agents

## Router (`llm/router.py`, `configs/models.yaml`)

* Providers: `ollama` (local), `groq`, `openrouter`, `proprietary` (any OpenAI-compatible endpoint).
  Model IDs are configuration only — check availability with `python scripts/check_providers.py [--ping]`.
* Tasks: `parser`, `optimizer`, `explainer` (its prompt quotes the resume as `[C#]` evidence) and `tiebreak`
  (resolves resume skill mentions) are `pii: true` → local providers only (unless `privacy.allow_remote_pii: true`).
  `research` and `spo` receive job-posting text only and may fall back to hosted models.
* Each call is logged to SQLite `llm_calls`: provider, model, task, timestamp, latency, input/output tokens,
  cost (provider-reported or from `pricing`; **null when unknown**, never 0), success, error, retries.
* `complete_json` validates against a Pydantic schema with exactly one repair retry.

## Ontology-constrained extraction (`parsing/llm_parser.py`)

~350-word chunks, 10% overlap; fixed ontology + JSON shape (`llm/prompts/extraction.py`); every item must
carry a verbatim `evidence` quote. Items are kept only if the quote is found in the chunk **and** the value
occurs in the quote (`rejected_ungrounded`, `rejected_value_not_in_evidence` are counted). Results are cached
by hash(provider/model/prompt version/chunk).

## Prompt-injection defence

Document text is placed only in the user turn, inside `<document>` tags, with delimiter tags and
instruction-like phrases neutralised (`common/privacy.py`). The system prompt tells the model the content is
data. Detected phrases are surfaced to the user and the critic. Agents have no tools that act externally.

## Explanations (`explanations/`)

* Template: "Matched: X [C1][J1]", "Missing (required): Y [J2]", "Transferable: A is related to B (inferred) [G1]".
* LLM: citation-constrained prompt over the numbered evidence registry.
* Verifier: strict skill spotting (list-context rules relaxed), numbers must appear in cited items (or be
  computed counts), qualification claims ("proficient in…") need candidate evidence, invalid IDs are
  flagged; unsupported sentences are removed before display. Entity names (job titles, companies,
  occupation titles) are masked so they are not mistaken for skill claims.

## Agents (`agents/`)

Parser, KG-Build, Matching, Profile Optimizer, Research & Outreach, Scheduler, Critic — wired in
`agents/graph.py`. The optimizer only suggests adding context/numbers you can verify and reports missing
skills as gaps ("add only if true"); optional bullet rewrites are rejected if they introduce new skills or
numbers. Outreach drafts and reminders require user approval; nothing is sent.

## Local setup (Ollama)

```bash
brew install ollama && ollama serve          # or the macOS app
ollama pull qwen2.5:3b-instruct              # ~2 GB; fits in 8 GB RAM
python scripts/check_providers.py --ping
```
With Ollama running, uploads use LLM extraction automatically (`parsing.use_llm: auto`).

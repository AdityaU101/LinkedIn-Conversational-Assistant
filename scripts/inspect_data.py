"""Inspect every dataset under Data/raw and write manifests.

Outputs (all values computed from the actual files — nothing is assumed):
  Data/manifests/dataset_manifest.json
  Data/manifests/dataset_manifest.md
  Data/manifests/data_quality_report.md

Usage:  python scripts/inspect_data.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import PROJECT_ROOT, get_settings  # noqa: E402
from common.text import content_hash, normalize_key, strip_html  # noqa: E402

RAW = get_settings().path("raw")
OUT = get_settings().path("manifests")
CHUNK = 50_000


def rel(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT))


def human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def profile_csv(path: Path, id_cols: list[str] | None = None, keep: dict | None = None) -> dict:
    """Chunked profile: rows, columns, nulls, duplicate IDs. ``keep`` collects values."""
    rows = 0
    nulls: Counter = Counter()
    empty: Counter = Counter()
    cols: list[str] = []
    dtypes: dict[str, str] = {}
    id_seen: dict[str, set] = {c: set() for c in (id_cols or [])}
    id_dups: Counter = Counter()
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        if not cols:
            cols = list(chunk.columns)
            dtypes = {c: str(t) for c, t in chunk.dtypes.items()}
        rows += len(chunk)
        nulls.update(chunk.isna().sum().to_dict())
        for c in chunk.columns:
            if chunk[c].dtype == object or str(chunk[c].dtype).startswith("str"):
                empty[c] += int((chunk[c].astype("string").str.strip() == "").sum())
        for c in id_seen:
            vals = chunk[c].dropna().tolist()
            for v in vals:
                if v in id_seen[c]:
                    id_dups[c] += 1
                else:
                    id_seen[c].add(v)
        if keep is not None:
            keep["__hook__"](chunk)
    full_row_dups = None
    if path.stat().st_size < 60_000_000:
        full_row_dups = int(pd.read_csv(path, low_memory=False).duplicated().sum())
    return {
        "file": rel(path),
        "format": "csv",
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "columns": cols,
        "dtypes": dtypes,
        "missing": {c: {"null": int(nulls[c]), "empty_string": int(empty[c]),
                        "pct_missing": round(100 * (nulls[c] + empty[c]) / rows, 2) if rows else 0}
                    for c in cols},
        "id_columns": {c: {"unique": len(id_seen[c]), "duplicate_values": int(id_dups[c])} for c in id_seen},
        "duplicate_full_rows": full_row_dups,
        "_ids": id_seen,
    }


def strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# Dataset inspectors
# --------------------------------------------------------------------------- #
def inspect_linkedin(overlap: dict) -> dict:
    base = RAW / "linkedin_jobs"
    keys, hashes, urls = set(), set(), set()

    def hook(chunk: pd.DataFrame) -> None:
        for t, c, d in zip(chunk["title"], chunk["company_name"], chunk["description"]):
            keys.add((normalize_key(t), normalize_key(c)))
            if isinstance(d, str):
                hashes.add(content_hash(d))

    postings = profile_csv(base / "postings.csv", ["job_id"], {"__hook__": hook})
    overlap["linkedin_kaggle"] = {"keys": keys, "hashes": hashes}
    files = [postings]
    layout = {
        "companies/companies.csv": ["company_id"], "companies/company_industries.csv": [],
        "companies/company_specialities.csv": [], "companies/employee_counts.csv": [],
        "jobs/benefits.csv": [], "jobs/job_industries.csv": [], "jobs/job_skills.csv": [],
        "jobs/salaries.csv": ["salary_id"], "mappings/industries.csv": ["industry_id"],
        "mappings/skills.csv": ["skill_abr"],
    }
    profiles = {"postings.csv": postings}
    for f, ids in layout.items():
        profiles[f] = profile_csv(base / f, ids)
        files.append(profiles[f])

    job_ids = postings["_ids"]["job_id"]
    company_ids = profiles["companies/companies.csv"]["_ids"]["company_id"]
    fk = {}

    def coverage(f: str, col: str, ref: set, name: str) -> None:
        s = pd.read_csv(base / f, usecols=[col])[col].dropna()
        fk[name] = {"rows": int(len(s)), "matched": int(s.isin(ref).sum()),
                    "pct": round(100 * float(s.isin(ref).mean()), 2) if len(s) else None}

    for f in ["jobs/benefits.csv", "jobs/job_industries.csv", "jobs/job_skills.csv", "jobs/salaries.csv"]:
        coverage(f, "job_id", job_ids, f"{f}.job_id -> postings.job_id")
    for f in ["companies/company_industries.csv", "companies/company_specialities.csv", "companies/employee_counts.csv"]:
        coverage(f, "company_id", company_ids, f"{f}.company_id -> companies.company_id")
    coverage("postings.csv", "company_id", company_ids, "postings.company_id -> companies.company_id")
    coverage("jobs/job_skills.csv", "skill_abr", profiles["mappings/skills.csv"]["_ids"]["skill_abr"],
             "job_skills.skill_abr -> mappings/skills.skill_abr")
    coverage("jobs/job_industries.csv", "industry_id", profiles["mappings/industries.csv"]["_ids"]["industry_id"],
             "job_industries.industry_id -> mappings/industries.industry_id")

    flat_dups = sorted(p.name for p in base.glob("*.csv") if p.name != "postings.csv")
    lt = pd.read_csv(base / "postings.csv", usecols=["listed_time"])["listed_time"].dropna()
    return {
        "dataset_id": "linkedin_kaggle",
        "name": "LinkedIn Job Postings 2023–2024 (Kaggle, arshkon/linkedin-job-postings)",
        "source": "Kaggle download; original copies at ./postings.csv and ./archive/",
        "location": rel(base),
        "files": [strip_private(f) for f in files],
        "relationships": fk,
        "date_range": {
            "listed_time_min": datetime.fromtimestamp(lt.min() / 1000, timezone.utc).date().isoformat(),
            "listed_time_max": datetime.fromtimestamp(lt.max() / 1000, timezone.utc).date().isoformat(),
        },
        "notes": [
            "mappings/skills.csv holds 35 coarse LinkedIn job-function categories (e.g. 'Information Technology'), "
            "not fine-grained skills; they are stored as job function tags, not as normalized skills.",
            f"Byte-identical flat copies of the nested files also exist in {rel(base)} ({', '.join(flat_dups)}); "
            "loaders read only the nested companies/ jobs/ mappings/ layout.",
            "./postings.csv, ./archive/postings.csv and Data/raw/linkedin_jobs/postings.csv are identical (same size and MD5).",
        ],
        "temporal": "historical",
        "usage": ["retrieval corpus", "job parsing", "graph construction", "ranking evaluation pool"],
        "privacy": "Public job postings; contains company names and recruiter-written text, no candidate PII.",
    }


def inspect_datastax(overlap: dict) -> dict:
    path = RAW / "datastax_linkedin" / "postings.csv"
    keys, hashes = set(), set()

    def hook(chunk: pd.DataFrame) -> None:
        for t, c, d in zip(chunk["title"], chunk["company"], chunk["description"]):
            keys.add((normalize_key(t), normalize_key(c)))
            if isinstance(d, str):
                hashes.add(content_hash(d))

    prof = profile_csv(path, ["id", "link"], {"__hook__": hook})
    df = pd.read_csv(path, usecols=["source", "date_posted"])
    overlap["datastax_linkedin"] = {"keys": keys, "hashes": hashes}
    return {
        "dataset_id": "datastax_linkedin",
        "name": "DataStax LinkedIn Job Listings",
        "source": "Hugging Face (DataStax); original copy at ./clean_jobs.csv with ./gitattributes",
        "location": rel(path.parent),
        "files": [strip_private(prof)],
        "value_counts": {"source": df["source"].value_counts().head(10).to_dict()},
        "date_range": {"date_posted_min": str(df["date_posted"].min()), "date_posted_max": str(df["date_posted"].max())},
        "notes": ["Original filename in the workspace was clean_jobs.csv; stored here as postings.csv "
                  "next to its .gitattributes, separate from the Kaggle postings.csv.",
                  "work_type and employment_type columns are entirely/mostly null — see missing statistics."],
        "temporal": "historical",
        "usage": ["retrieval corpus", "job parsing"],
        "privacy": "Public job postings.",
    }


def inspect_insights(overlap: dict) -> dict:
    path = RAW / "linkedin_job_insights" / "linkedin_job_posts_insights.xlsx"
    xls = pd.ExcelFile(path)
    sheets = {}
    keys = set()
    for s in xls.sheet_names:
        df = xls.parse(s)
        sheets[s] = {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing": {c: int(df[c].isna().sum()) for c in df.columns},
            "duplicate_full_rows": int(df.duplicated().sum()),
            "value_counts": {c: df[c].value_counts().head(8).astype(int).to_dict()
                             for c in ["seniority_level", "employment_type", "hiring_status"] if c in df},
        }
        if "date" in df:
            sheets[s]["date_range"] = [str(df["date"].min()), str(df["date"].max())]
        for t, c in zip(df.get("job_title", []), df.get("company_name", [])):
            keys.add((normalize_key(t), normalize_key(c)))
    overlap["linkedin_insights"] = {"keys": keys, "hashes": set()}
    return {
        "dataset_id": "linkedin_insights",
        "name": "LinkedIn Job Posts Insights (XLSX)",
        "source": "Workspace file ./linkedin_job_posts_insights.xlsx",
        "location": rel(path.parent),
        "files": [{"file": rel(path), "format": "xlsx", "size_bytes": path.stat().st_size, "sheets": sheets}],
        "notes": ["No job description or URL column: usable for market analytics and dedup cross-reference, "
                  "not for text retrieval. Excluded from the recommendation index by default."],
        "temporal": "historical",
        "usage": ["analytics", "cross-source dedup"],
        "privacy": "Public posting metadata.",
    }


def inspect_kaggle_resumes() -> dict:
    base = RAW / "resume_dataset"
    prof = profile_csv(base / "Resume.csv", ["ID"])
    df = pd.read_csv(base / "Resume.csv", usecols=["ID", "Category"])
    pdfs = list((base / "data").rglob("*.pdf"))
    pdf_ids = {int(p.stem) for p in pdfs if p.stem.isdigit()}
    cats = sorted(p.name for p in (base / "data").iterdir() if p.is_dir())
    pdf_cat = {(p.parent.name, int(p.stem)) for p in pdfs if p.stem.isdigit()}
    csv_cat = set(zip(df["Category"], df["ID"]))
    return {
        "dataset_id": "kaggle_resumes",
        "name": "Kaggle Resume Dataset (snehaanbhawal/resume-dataset)",
        "source": "Workspace ./archive-2/",
        "location": rel(base),
        "files": [strip_private(prof),
                  {"file": rel(base / "data"), "format": "pdf", "count": len(pdfs),
                   "size_bytes": sum(p.stat().st_size for p in pdfs)}],
        "categories": {"from_folders": cats, "count": len(cats),
                       "csv_distribution": df["Category"].value_counts().to_dict()},
        "relationships": {
            "csv_ids_with_pdf": int(df["ID"].isin(pdf_ids).sum()),
            "csv_rows": int(len(df)),
            "category_agreement_csv_vs_folder": len(csv_cat & pdf_cat),
        },
        "temporal": "historical",
        "usage": ["candidate pool for job->candidate matching", "resume parsing evaluation pool",
                  "occupation-family classifier (real category labels)", "annotation sampling"],
        "privacy": "Resumes are anonymised by the publisher (names/companies replaced), but may still contain "
                   "quasi-identifiers. Treated as sensitive: never sent to hosted LLMs by default.",
    }


def _corpus_records(zpath: Path, member: str, limit: int | None = None):
    with zipfile.ZipFile(zpath) as zf, zf.open(member) as fh:
        for i, raw in enumerate(fh):
            line = raw.decode("utf-8", "replace").strip()
            if not line or set(line) == {":"}:
                continue
            parts = line.split(":::", 2)
            if len(parts) == 3:
                yield parts
            if limit and i > limit:
                break


def inspect_resume_corpus() -> dict:
    base = RAW / "resume_corpus"
    files = []
    for p in sorted(base.iterdir()):
        files.append({"file": rel(p), "format": p.suffix.lstrip(".") or "txt", "size_bytes": p.stat().st_size})
    samples = list(_corpus_records(base / "resume_samples.zip", "resume_samples.txt"))
    label_counts: Counter = Counter()
    for _, labels, _ in samples:
        label_counts.update(l for l in labels.split(";") if l)
    with zipfile.ZipFile(base / "resumes_corpus.zip") as zf:
        names = zf.namelist()
    txt = [n for n in names if n.endswith(".txt")]
    lab = [n for n in names if n.endswith(".lab")]
    with zipfile.ZipFile(base / "resumes_corpus.zip") as zf:
        lab_counts: Counter = Counter()
        for n in lab:
            lab_counts.update(l.strip() for l in zf.read(n).decode("utf-8", "replace").splitlines() if l.strip())
    classes = (base / "normlized_classes.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    norm_targets: Counter = Counter()
    for line in classes:
        if ":" in line:
            for t in line.rsplit(":", 1)[1].split(","):
                norm_targets[t.strip()] += 1
    it_lines = (base / "skills_it.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    it_records = [l for l in it_lines if ":::" in l and set(l) != {":"}]
    return {
        "dataset_id": "resume_corpus",
        "name": "resume_corpus (Jiechieu & Tsopze, 2020) — multi-labelled IT resumes",
        "source": "Workspace ./resume_corpus-master/ (GitHub repository snapshot)",
        "location": rel(base),
        "files": files,
        "structure": {
            "resume_samples.zip": {"member": "resume_samples.txt", "records": len(samples),
                                   "format": "id:::occupation labels (';'-separated):::resume text, records separated by '::::::' lines",
                                   "distinct_raw_labels": len(label_counts),
                                   "top_raw_labels": dict(label_counts.most_common(10))},
            "resumes_corpus.zip": {"txt_files": len(txt), "lab_files": len(lab),
                                   "label_distribution": dict(lab_counts.most_common())},
            "normlized_classes.txt": {"lines": len(classes), "format": "raw occupation title:normalized class[,class]",
                                      "normalized_classes": dict(norm_targets.most_common())},
            "skills_it.txt": {"records": len(it_records),
                              "format": "Same id:::labels:::text layout as resume_samples.txt (it is a resume "
                                        "subset, not a skill list, despite the filename)"},
        },
        "notes": ["The README calls the archive 'resumes_sample.zip'; the actual file is resume_samples.zip.",
                  "The normalized classes file is spelled 'normlized_classes.txt' in the repository."],
        "temporal": "historical",
        "usage": ["occupation-normalization evaluation (expert raw-title -> class mappings)",
                  "candidate pool", "classical extraction training text"],
        "privacy": "Real resumes scraped from public job boards; contain locations and employer names. Treated as sensitive.",
    }


def inspect_onet() -> dict:
    base = RAW / "onet"
    files = []
    for p in sorted(base.glob("*.csv")):
        df = pd.read_csv(p, low_memory=False)
        files.append({"file": rel(p), "format": "csv", "size_bytes": p.stat().st_size, "rows": len(df),
                      "columns": list(df.columns),
                      "missing_total": int(df.isna().sum().sum()),
                      "distinct_occupations": int(df["O*NET-SOC Code"].nunique()) if "O*NET-SOC Code" in df else None})
    readme = (base / "Read Me.txt").read_text(errors="replace").splitlines()[:3]
    occ = pd.read_csv(base / "occupation_data.csv")
    sw = pd.read_csv(base / "software_skills.csv")
    return {
        "dataset_id": "onet",
        "name": "O*NET Database " + " ".join(l.strip() for l in readme if l.strip()),
        "source": "Workspace ./db_31_0_csv/ (onetcenter.org, CC BY 4.0)",
        "location": rel(base),
        "files": files,
        "identifiers": {"occupation": "O*NET-SOC Code (e.g. 15-2051.00)", "element": "Element ID (content model)",
                        "task": "Task ID", "dwa": "DWA Element ID"},
        "summary": {"occupations": int(occ["O*NET-SOC Code"].nunique()),
                    "software_skill_rows": len(sw),
                    "distinct_software_examples": int(sw["Workplace Example"].nunique()),
                    "software_categories": int(sw["Element ID"].nunique()),
                    "hot_technology_examples": int(sw.loc[sw["Hot Technology"] == "Y", "Workplace Example"].nunique())},
        "relationships": {"*.O*NET-SOC Code -> occupation_data.O*NET-SOC Code": "all occupation-level files",
                          "*.Element ID -> content_model_reference.Element ID": "skills/knowledge/abilities/activities",
                          "related_occupations.Related O*NET-SOC Code -> occupation_data": "occupation graph"},
        "notes": ["Release 31.0 uses new file names: essential_skills.csv and transferable_skills.csv "
                  "(the former 'Skills' file split), software_skills.csv (former Technology Skills).",
                  "software_skills Element IDs identify technology categories; individual tools (Workplace Example) "
                  "have no O*NET ID, so the project derives stable IDs 'onet-tech:<slug>'."],
        "temporal": "reference taxonomy (2026 release)",
        "usage": ["primary skill taxonomy", "occupation normalization", "transferable skills", "occupation relations"],
        "privacy": "Public reference data.",
    }


def inspect_osmt() -> dict:
    base = RAW / "osmt"
    files = []
    for p in sorted((base / "import").glob("*.csv")):
        df = pd.read_csv(p)
        files.append({"file": rel(p), "format": "csv", "size_bytes": p.stat().st_size, "rows": len(df),
                      "columns": list(df.columns)})
    return {
        "dataset_id": "osmt",
        "name": "Open Skills Management Tool (OSMT) — import resources",
        "source": "Workspace ./osmt-develop/ (WGU OSMT source repository). Only import/ CSVs, README and "
                  "LICENSE were copied; the repository (incl. 1.2 GB UI sources) is preserved in place.",
        "location": rel(base),
        "files": files,
        "notes": ["OSMT is application source code (Kotlin/Angular), not a dataset. Its import/ folder contains "
                  "a BLS SOC hierarchy (BLS-Import.csv), an O*NET occupation list identical in structure to "
                  "occupation_data.csv, a 2-row example Rich Skill Descriptor file, and a batch-import template.",
                  "Used: BLS-Import.csv for SOC major/minor group names; RSD columns inform the Skill node schema."],
        "temporal": "reference",
        "usage": ["SOC hierarchy names", "skill schema reference"],
        "privacy": "Public.",
    }


def inspect_remoteok(overlap: dict) -> dict:
    base = RAW / "remoteok"
    files, keys, hashes = [], set(), set()
    total, legal = 0, None
    for p in sorted(base.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        jobs = [d for d in data if isinstance(d, dict) and "id" in d]
        legal = legal or next((d.get("legal") for d in data if isinstance(d, dict) and "legal" in d), None)
        total += len(jobs)
        fields = Counter(k for j in jobs for k in j)
        files.append({"file": rel(p), "format": "json", "size_bytes": p.stat().st_size, "records": len(jobs),
                      "fields": dict(fields),
                      "dates": [min(j.get("date", "") for j in jobs), max(j.get("date", "") for j in jobs)] if jobs else None,
                      "duplicate_ids": len(jobs) - len({j["id"] for j in jobs})})
        for j in jobs:
            keys.add((normalize_key(j.get("position")), normalize_key(j.get("company"))))
            hashes.add(content_hash(strip_html(j.get("description", ""))))
    overlap["remoteok"] = {"keys": keys, "hashes": hashes}
    return {
        "dataset_id": "remoteok", "name": "RemoteOK public job feed (https://remoteok.com/api)",
        "source": "Downloaded API JSON", "location": rel(base), "files": files, "records": total,
        "terms": legal,
        "notes": ["First array element is the API legal notice, not a job.",
                  "Some text fields are double-encoded UTF-8 (mojibake); the loader repairs them without altering raw files.",
                  "Attribution required: link back to the RemoteOK job URL and mention Remote OK as the source."],
        "temporal": "live snapshot", "usage": ["retrieval corpus (current jobs)"], "privacy": "Public postings.",
    }


def inspect_adzuna(overlap: dict) -> dict:
    base = RAW / "adzuna"
    files = sorted(base.glob("*.json"))
    keys = set()
    n = 0
    for p in files:
        data = json.loads(p.read_text())
        for r in data.get("results", []):
            n += 1
            keys.add((normalize_key(r.get("title")), normalize_key((r.get("company") or {}).get("display_name"))))
    overlap["adzuna"] = {"keys": keys, "hashes": set()}
    return {
        "dataset_id": "adzuna", "name": "Adzuna Jobs API snapshots", "source": "https://api.adzuna.com (credentials via env)",
        "location": rel(base), "files": [{"file": rel(p), "format": "json", "size_bytes": p.stat().st_size} for p in files],
        "records": n,
        "notes": ["Populated only when ADZUNA_APP_ID/ADZUNA_APP_KEY are set and ingestion fetches live jobs."
                  if not files else "Raw API responses saved verbatim per fetch."],
        "temporal": "live snapshot", "usage": ["retrieval corpus (current jobs)"], "privacy": "Public postings.",
    }


def cross_source_overlap(overlap: dict) -> dict:
    names = list(overlap)
    out = {"title_company_key": {}, "description_hash": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ka, kb = overlap[a]["keys"], overlap[b]["keys"]
            ka = {k for k in ka if k[0]}
            kb = {k for k in kb if k[0]}
            out["title_company_key"][f"{a} ∩ {b}"] = len(ka & kb)
            ha, hb = overlap[a]["hashes"], overlap[b]["hashes"]
            if ha and hb:
                out["description_hash"][f"{a} ∩ {b}"] = len(ha & hb)
    out["sizes"] = {n: {"distinct_title_company": len(overlap[n]["keys"]),
                        "distinct_description_hash": len(overlap[n]["hashes"])} for n in names}
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_md(manifest: dict) -> str:
    lines = ["# Dataset Manifest", "", f"Generated {manifest['generated_at']} by `scripts/inspect_data.py`. "
             "All counts are computed from the files.", ""]
    for ds in manifest["datasets"]:
        lines += [f"## {ds['name']}", "", f"- **ID:** `{ds['dataset_id']}`", f"- **Location:** `{ds['location']}`",
                  f"- **Source:** {ds['source']}", f"- **Temporal:** {ds['temporal']}",
                  f"- **Usage:** {', '.join(ds['usage'])}", f"- **Privacy:** {ds['privacy']}", "",
                  "| File | Format | Size | Rows/records |", "|---|---|---|---|"]
        for f in ds["files"]:
            rows = f.get("rows", f.get("records", f.get("count", "")))
            if "sheets" in f:
                rows = ", ".join(f"{s}: {v['rows']}" for s, v in f["sheets"].items())
            lines.append(f"| `{f['file']}` | {f.get('format','')} | {human(f.get('size_bytes', 0))} | {rows} |")
        lines.append("")
        for f in ds["files"]:
            if f.get("columns"):
                lines.append(f"**`{Path(f['file']).name}` columns:** " + ", ".join(f"`{c}`" for c in f["columns"]))
                lines.append("")
        if ds.get("relationships"):
            lines += ["**Relationships / FK coverage:**", "", "```json", json.dumps(ds["relationships"], indent=1, default=str), "```", ""]
        if ds.get("structure"):
            lines += ["**Structure:**", "", "```json", json.dumps(ds["structure"], indent=1, default=str)[:4000], "```", ""]
        for n in ds.get("notes", []):
            lines.append(f"> {n}")
        lines.append("")
    lines += ["## Cross-source overlap", "", "```json", json.dumps(manifest["cross_source_overlap"], indent=1), "```"]
    return "\n".join(lines)


def render_quality(manifest: dict) -> str:
    lines = ["# Data Quality Report", "", f"Generated {manifest['generated_at']}.", "",
             "Columns with ≥ 20% missing values and duplicate statistics per file.", ""]
    for ds in manifest["datasets"]:
        lines.append(f"## {ds['dataset_id']}")
        for f in ds["files"]:
            miss = f.get("missing")
            if isinstance(miss, dict) and miss and isinstance(next(iter(miss.values())), dict):
                bad = {c: v["pct_missing"] for c, v in miss.items() if v["pct_missing"] >= 20}
                lines.append(f"- `{Path(f['file']).name}`: {f['rows']} rows; duplicate full rows: "
                             f"{f.get('duplicate_full_rows')}; ID duplicates: "
                             f"{ {c: v['duplicate_values'] for c, v in f.get('id_columns', {}).items()} }")
                if bad:
                    lines.append("  - high-missing columns (%): " + ", ".join(f"`{c}` {p}" for c, p in sorted(bad.items(), key=lambda x: -x[1])))
            if "sheets" in f:
                for s, v in f["sheets"].items():
                    lines.append(f"- `{Path(f['file']).name}` [{s}]: {v['rows']} rows; duplicate rows {v['duplicate_full_rows']}; "
                                 f"missing {v['missing']}")
            if "duplicate_ids" in f:
                lines.append(f"- `{Path(f['file']).name}`: {f['records']} records, duplicate ids {f['duplicate_ids']}")
        lines.append("")
    ov = manifest["cross_source_overlap"]
    lines += ["## Cross-source duplicates", "",
              "Exact matches on normalised (title, company) and on normalised description hash. "
              "These sources are therefore **not** independent; the ingestion pipeline deduplicates "
              "across sources (exact + MinHash) and assigns near-duplicate clusters to a single split.", "",
              "| Pair | title+company matches | description-hash matches |", "|---|---|---|"]
    for pair, n in ov["title_company_key"].items():
        lines.append(f"| {pair} | {n} | {ov['description_hash'].get(pair, 'n/a')} |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    overlap: dict = {}
    datasets = []
    for fn in (inspect_linkedin, inspect_datastax, inspect_insights, inspect_remoteok, inspect_adzuna):
        print("inspecting", fn.__name__, flush=True)
        datasets.append(fn(overlap))
    for fn in (inspect_kaggle_resumes, inspect_resume_corpus, inspect_onet, inspect_osmt):
        print("inspecting", fn.__name__, flush=True)
        datasets.append(fn())
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "datasets": datasets, "cross_source_overlap": cross_source_overlap(overlap)}
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUT / "dataset_manifest.md").write_text(render_md(manifest))
    (OUT / "data_quality_report.md").write_text(render_quality(manifest))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

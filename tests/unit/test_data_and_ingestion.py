"""Data discovery, loaders (CSV / XLSX / ZIP / JSON), normalisation and deduplication."""
import json

import httpx
import pandas as pd
import pytest

from common.config import get_settings
from common.text import content_hash, find_span, fix_mojibake, strip_html, word_chunks
from ingestion import adzuna_loader, normalize as N, remoteok_loader
from ingestion.deduplicate import assign_splits, check_no_leakage, deduplicate

RAW = get_settings().path("raw")


# ---------------------------------------------------------------- text utils
def test_word_chunks_overlap_and_offsets():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = word_chunks(text, 350, 0.10)
    assert len(chunks) == 4                      # starts at words 0, 315, 630, 945
    assert chunks[-1][2].split()[-1] == "w999"    # nothing dropped at the end
    for s, e, c in chunks:
        assert text[s:e] == c
    # 10% overlap: second chunk starts 315 words in
    assert chunks[1][2].split()[0] == "w315"


def test_find_span_whitespace_insensitive():
    hay = "Built   models in\nPyTorch daily"
    assert find_span(hay, "models in PyTorch") == (8, 25)
    assert find_span(hay, "TensorFlow") is None


def test_mojibake_and_html():
    assert fix_mojibake("Telefondienst fÃ¼r Praxis") == "Telefondienst für Praxis"
    assert fix_mojibake("café") == "café"   # correct text untouched
    assert strip_html("<p>Hello&nbsp;<b>world</b></p><li>x</li>") == "Hello world\nx"


# ---------------------------------------------------------------- discovery
def test_manifest_matches_files():
    m = json.loads((get_settings().path("manifests") / "dataset_manifest.json").read_text())
    ids = {d["dataset_id"] for d in m["datasets"]}
    assert {"linkedin_kaggle", "datastax_linkedin", "linkedin_insights", "kaggle_resumes", "resume_corpus", "onet",
            "osmt", "remoteok"} <= ids
    li = next(d for d in m["datasets"] if d["dataset_id"] == "linkedin_kaggle")
    postings = next(f for f in li["files"] if f["file"].endswith("postings.csv"))
    assert postings["rows"] == 123_849
    assert postings["id_columns"]["job_id"]["duplicate_values"] == 0


def test_resume_categories_discovered_from_folders():
    from ingestion.resume_loader import categories, iter_resumes

    cats = categories()
    assert len(cats) == 24 and "INFORMATION-TECHNOLOGY" in cats
    r = next(iter_resumes(limit=1))
    assert r.text and r.category in cats and r.pdf_path and r.pdf_path.exists()


# ---------------------------------------------------------------- loaders
def test_linkedin_loader_csv_chunked_sample():
    from ingestion.linkedin_loader import LinkedInLoader

    jobs = list(LinkedInLoader(chunk_size=5000).iter_jobs(limit=40))
    assert len(jobs) == 40
    j = jobs[0]
    assert j.job_id.startswith("linkedin_kaggle:") and j.title and j.is_historical
    assert j.raw_source_reference.startswith("Data/raw/linkedin_jobs/postings.csv")
    for j in jobs:  # salary only when known; annualised
        if j.salary_min is not None:
            assert j.salary_currency and j.salary_min > 1000


def test_datastax_loader():
    from ingestion.datastax_loader import DataStaxLoader

    jobs = list(DataStaxLoader().iter_jobs(limit=10))
    assert jobs[0].source == "datastax_linkedin" and jobs[0].source_url.startswith("https://")
    assert all(j.employment_type is None for j in jobs)  # column is 100% null in the source: never invented


def test_insights_loader_xlsx():
    from ingestion.insights_loader import InsightsLoader

    jobs = list(InsightsLoader().iter_jobs(limit=25))
    assert len(jobs) == 25 and all(j.description is None for j in jobs)
    assert jobs[0].raw_source_reference.endswith("Sheet1-2")


def test_resume_corpus_zip_loader():
    from ingestion.resume_corpus_loader import iter_corpus, iter_samples, normalized_classes

    s = next(iter_samples(limit=1))
    assert s.raw_labels and len(s.text) > 100
    c = next(iter_corpus(limit=1))
    assert c.labels and c.text
    nc = normalized_classes()
    assert len(nc) > 19_000 and all(cls for _, cls in nc[:50])


def test_onet_loader_ids():
    from ingestion.onet_loader import ONetLoader

    o = ONetLoader()
    occ = o.occupations()
    assert len(occ) == 1016 and occ["code"].str.match(r"^\d{2}-\d{4}\.\d{2}$").all()
    tech = o.technology_skills()
    assert tech["skill_id"].is_unique and "PyTorch" in set(tech["name"])


def test_remoteok_record_and_fetch_mocked(tmp_path):
    rec = {"id": "42", "position": "Backend Engineer", "company": "Acme", "tags": "['python', 'aws']",
           "description": "<p>Build APIs</p>", "location": "Worldwide", "salary_min": 0, "salary_max": 0,
           "date": "2026-10-01T00:00:00+00:00", "url": "https://remoteOK.com/remote-jobs/42"}
    job = remoteok_loader.record_to_job(rec, "f.json")
    assert job.work_mode == "remote" and job.skills == ["python", "aws"] and job.salary_min is None
    assert job.attribution.startswith("Remote OK") and job.source_url.endswith("/42")

    def handler(request):
        return httpx.Response(200, json=[{"legal": "terms"}, rec])

    path = remoteok_loader.fetch(httpx.Client(transport=httpx.MockTransport(handler)), out_dir=tmp_path)
    jobs = list(remoteok_loader.RemoteOKLoader(tmp_path).iter_jobs())
    assert path.exists() and len(jobs) == 1


def test_adzuna_requires_credentials_and_mocked_fetch(tmp_path, monkeypatch):
    assert not adzuna_loader.available()
    with pytest.raises(RuntimeError):
        adzuna_loader.fetch()
    monkeypatch.setenv("ADZUNA_APP_ID", "id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "secret-key")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"results": [
            {"id": 1, "title": "Data Engineer", "company": {"display_name": "X"}, "location": {"display_name": "Austin, TX"},
             "description": "Spark", "salary_min": 100000, "salary_max": 120000, "salary_is_predicted": "1",
             "created": "2026-09-30T10:00:00Z", "redirect_url": "https://adzuna.example/1"}]})

    files = adzuna_loader.fetch(what="data", client=httpx.Client(transport=httpx.MockTransport(handler)), out_dir=tmp_path)
    assert "secret-key" not in files[0].read_text()  # the key is never persisted
    job = next(adzuna_loader.AdzunaLoader(tmp_path).iter_jobs())
    assert job.salary_min is None  # predicted salaries are not presented as posted values
    assert job.location_state == "TX"


# ---------------------------------------------------------------- normalisation
def test_field_normalisers():
    assert N.parse_location("Princeton, NJ")["state"] == "NJ"
    assert N.parse_location("Brisbane, Queensland, Australia")["country"] == "Australia"
    assert N.annualize_salary(20, "HOURLY") == 41600
    assert N.annualize_salary(5000, "MONTHLY") == 60000
    assert N.annualize_salary(None, "HOURLY") is None and N.annualize_salary(10, None) is None
    assert N.normalize_employment_type("Full-time") == "full_time"
    assert N.normalize_seniority("Mid-Senior level") == "mid_senior"
    assert N.infer_work_mode("Engineer (Hybrid)") == "hybrid"
    assert N.infer_work_mode("Engineer", remote_flag=1.0) == "remote"
    assert N.infer_work_mode("Engineer") is None  # no signal -> unknown
    assert N.parse_date("2025-04-14") == "2025-04-14"


# ---------------------------------------------------------------- dedup
def _df(rows):
    base = {"source": "linkedin_kaggle", "title": "t", "company": "c", "location": "l", "description": None}
    out = [{**base, **r} for r in rows]
    for r in out:
        r["content_hash"] = content_hash(r["description"] or r["title"])
    return pd.DataFrame(out)


def test_dedup_signals_and_cluster_splits():
    long = "We are hiring a data engineer to build pipelines with Spark and Airflow in the cloud. " * 4
    near = long.replace("cloud.", "cloud!", 1) + " Benefits included."
    df = _df([
        {"job_id": "a:1", "title": "Data Engineer", "company": "X", "description": long},
        {"job_id": "b:1", "source": "datastax_linkedin", "title": "Data Engineer II", "company": "Y", "description": long},
        {"job_id": "a:2", "title": "Other", "company": "Z", "description": near},
        {"job_id": "a:3", "title": "Nurse", "company": "H", "location": "TX", "description": None},
        {"job_id": "i:1", "source": "linkedin_insights", "title": "Nurse", "company": "H", "location": "TX"},
        {"job_id": "a:4", "title": "Chef", "company": "R", "description": "Cook food " * 30},
    ])
    out, rep = deduplicate(df, threshold=0.8)
    cl = dict(zip(out["job_id"], out["dedup_cluster"]))
    assert cl["a:1"] == cl["b:1"] == cl["a:2"]          # hash + MinHash near-duplicate
    assert cl["a:3"] == cl["i:1"]                        # title/company/location
    assert cl["a:4"] not in {cl["a:1"], cl["a:3"]}
    assert rep["duplicate_records"] == 3 and "minhash_near_duplicate" in rep["merges_by_signal"]
    assert out.groupby("dedup_cluster")["is_representative"].sum().eq(1).all()
    out["split"] = assign_splits(out["dedup_cluster"])
    assert check_no_leakage(out) is None


def test_missing_descriptions_never_merge_by_hash():
    df = _df([{"job_id": "x:1", "title": "A", "company": "1"}, {"job_id": "x:2", "title": "B", "company": "2"}])
    out, rep = deduplicate(df, use_minhash=False)
    assert out["dedup_cluster"].nunique() == 2


def test_source_masked_values_are_unknown_not_values():
    from ingestion.normalize import is_masked

    assert is_masked("**** *********") and is_masked("*********, *********, *****") and is_masked("***")
    assert not is_masked("Data Scientist") and not is_masked("C*") and not is_masked("") and not is_masked(None)

# Data Quality Report

Generated 2026-10-09T01:19:10+00:00.

Columns with ≥ 20% missing values and duplicate statistics per file.

## linkedin_kaggle
- `postings.csv`: 123849 rows; duplicate full rows: None; ID duplicates: {'job_id': 0}
  - high-missing columns (%): `closed_time` 99.13, `skills_desc` 98.03, `med_salary` 94.93, `remote_allowed` 87.69, `applies` 81.17, `max_salary` 75.94, `min_salary` 75.94, `pay_period` 70.87, `currency` 70.87, `compensation_type` 70.87, `normalized_salary` 70.87, `posting_domain` 32.27, `application_url` 29.6, `formatted_experience_level` 23.75, `fips` 22.14
- `companies.csv`: 24473 rows; duplicate full rows: 0; ID duplicates: {'company_id': 0}
- `company_industries.csv`: 24375 rows; duplicate full rows: 0; ID duplicates: {}
- `company_specialities.csv`: 169387 rows; duplicate full rows: 0; ID duplicates: {}
- `employee_counts.csv`: 35787 rows; duplicate full rows: 0; ID duplicates: {}
- `benefits.csv`: 67943 rows; duplicate full rows: 0; ID duplicates: {}
- `job_industries.csv`: 164808 rows; duplicate full rows: 0; ID duplicates: {}
- `job_skills.csv`: 213768 rows; duplicate full rows: 0; ID duplicates: {}
- `salaries.csv`: 40785 rows; duplicate full rows: 0; ID duplicates: {'salary_id': 0}
  - high-missing columns (%): `med_salary` 83.23
- `industries.csv`: 422 rows; duplicate full rows: 0; ID duplicates: {'industry_id': 0}
- `skills.csv`: 35 rows; duplicate full rows: 0; ID duplicates: {'skill_abr': 0}

## datastax_linkedin
- `postings.csv`: 1048 rows; duplicate full rows: 0; ID duplicates: {'id': 0, 'link': 0}
  - high-missing columns (%): `work_type` 100.0, `employment_type` 100.0

## linkedin_insights
- `linkedin_job_posts_insights.xlsx` [Sheet1]: 31597 rows; duplicate rows 10693; missing {'job_title': 26, 'company_name': 940, 'location': 9, 'hiring_status': 0, 'date': 0, 'seniority_level': 1308, 'job_function': 1590, 'employment_type': 1591, 'industry': 2011}

## remoteok
- `remoteok_jobs.json`: 99 records, duplicate ids 0

## adzuna

## kaggle_resumes
- `Resume.csv`: 2484 rows; duplicate full rows: 0; ID duplicates: {'ID': 0}

## resume_corpus

## onet

## osmt

## Cross-source duplicates

Exact matches on normalised (title, company) and on normalised description hash. These sources are therefore **not** independent; the ingestion pipeline deduplicates across sources (exact + MinHash) and assigns near-duplicate clusters to a single split.

| Pair | title+company matches | description-hash matches |
|---|---|---|
| linkedin_kaggle ∩ datastax_linkedin | 14 | 0 |
| linkedin_kaggle ∩ linkedin_insights | 58 | n/a |
| linkedin_kaggle ∩ remoteok | 0 | 0 |
| linkedin_kaggle ∩ adzuna | 2 | n/a |
| datastax_linkedin ∩ linkedin_insights | 5 | n/a |
| datastax_linkedin ∩ remoteok | 0 | 0 |
| datastax_linkedin ∩ adzuna | 0 | n/a |
| linkedin_insights ∩ remoteok | 0 | n/a |
| linkedin_insights ∩ adzuna | 2 | n/a |
| remoteok ∩ adzuna | 0 | n/a |
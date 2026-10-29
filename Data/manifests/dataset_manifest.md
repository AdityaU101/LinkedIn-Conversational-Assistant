# Dataset Manifest

Generated 2026-10-09T01:19:10+00:00 by `scripts/inspect_data.py`. All counts are computed from the files.

## LinkedIn Job Postings 2023–2024 (Kaggle, arshkon/linkedin-job-postings)

- **ID:** `linkedin_kaggle`
- **Location:** `Data/raw/linkedin_jobs`
- **Source:** Kaggle download; original copies at ./postings.csv and ./archive/
- **Temporal:** historical
- **Usage:** retrieval corpus, job parsing, graph construction, ranking evaluation pool
- **Privacy:** Public job postings; contains company names and recruiter-written text, no candidate PII.

| File | Format | Size | Rows/records |
|---|---|---|---|
| `Data/raw/linkedin_jobs/postings.csv` | csv | 492.9 MB | 123849 |
| `Data/raw/linkedin_jobs/companies/companies.csv` | csv | 22.1 MB | 24473 |
| `Data/raw/linkedin_jobs/companies/company_industries.csv` | csv | 764.1 KB | 24375 |
| `Data/raw/linkedin_jobs/companies/company_specialities.csv` | csv | 4.2 MB | 169387 |
| `Data/raw/linkedin_jobs/companies/employee_counts.csv` | csv | 1020.1 KB | 35787 |
| `Data/raw/linkedin_jobs/jobs/benefits.csv` | csv | 1.8 MB | 67943 |
| `Data/raw/linkedin_jobs/jobs/job_industries.csv` | csv | 2.4 MB | 164808 |
| `Data/raw/linkedin_jobs/jobs/job_skills.csv` | csv | 3.3 MB | 213768 |
| `Data/raw/linkedin_jobs/jobs/salaries.csv` | csv | 2.1 MB | 40785 |
| `Data/raw/linkedin_jobs/mappings/industries.csv` | csv | 12.4 KB | 422 |
| `Data/raw/linkedin_jobs/mappings/skills.csv` | csv | 679.0 B | 35 |

**`postings.csv` columns:** `job_id`, `company_name`, `title`, `description`, `max_salary`, `pay_period`, `location`, `company_id`, `views`, `med_salary`, `min_salary`, `formatted_work_type`, `applies`, `original_listed_time`, `remote_allowed`, `job_posting_url`, `application_url`, `application_type`, `expiry`, `closed_time`, `formatted_experience_level`, `skills_desc`, `listed_time`, `posting_domain`, `sponsored`, `work_type`, `currency`, `compensation_type`, `normalized_salary`, `zip_code`, `fips`

**`companies.csv` columns:** `company_id`, `name`, `description`, `company_size`, `state`, `country`, `city`, `zip_code`, `address`, `url`

**`company_industries.csv` columns:** `company_id`, `industry`

**`company_specialities.csv` columns:** `company_id`, `speciality`

**`employee_counts.csv` columns:** `company_id`, `employee_count`, `follower_count`, `time_recorded`

**`benefits.csv` columns:** `job_id`, `inferred`, `type`

**`job_industries.csv` columns:** `job_id`, `industry_id`

**`job_skills.csv` columns:** `job_id`, `skill_abr`

**`salaries.csv` columns:** `salary_id`, `job_id`, `max_salary`, `med_salary`, `min_salary`, `pay_period`, `currency`, `compensation_type`

**`industries.csv` columns:** `industry_id`, `industry_name`

**`skills.csv` columns:** `skill_abr`, `skill_name`

**Relationships / FK coverage:**

```json
{
 "jobs/benefits.csv.job_id -> postings.job_id": {
  "rows": 67943,
  "matched": 65393,
  "pct": 96.25
 },
 "jobs/job_industries.csv.job_id -> postings.job_id": {
  "rows": 164808,
  "matched": 158963,
  "pct": 96.45
 },
 "jobs/job_skills.csv.job_id -> postings.job_id": {
  "rows": 213768,
  "matched": 205778,
  "pct": 96.26
 },
 "jobs/salaries.csv.job_id -> postings.job_id": {
  "rows": 40785,
  "matched": 36073,
  "pct": 88.45
 },
 "companies/company_industries.csv.company_id -> companies.company_id": {
  "rows": 24375,
  "matched": 24375,
  "pct": 100.0
 },
 "companies/company_specialities.csv.company_id -> companies.company_id": {
  "rows": 169387,
  "matched": 169387,
  "pct": 100.0
 },
 "companies/employee_counts.csv.company_id -> companies.company_id": {
  "rows": 35787,
  "matched": 35787,
  "pct": 100.0
 },
 "postings.company_id -> companies.company_id": {
  "rows": 122132,
  "matched": 122131,
  "pct": 100.0
 },
 "job_skills.skill_abr -> mappings/skills.skill_abr": {
  "rows": 213768,
  "matched": 213768,
  "pct": 100.0
 },
 "job_industries.industry_id -> mappings/industries.industry_id": {
  "rows": 164808,
  "matched": 164808,
  "pct": 100.0
 }
}
```

> mappings/skills.csv holds 35 coarse LinkedIn job-function categories (e.g. 'Information Technology'), not fine-grained skills; they are stored as job function tags, not as normalized skills.
> Byte-identical flat copies of the nested files also exist in Data/raw/linkedin_jobs (benefits.csv, companies.csv, company_industries.csv, company_specialities.csv, employee_counts.csv, industries.csv, job_industries.csv, job_skills.csv, salaries.csv, skills.csv); loaders read only the nested companies/ jobs/ mappings/ layout.
> ./postings.csv, ./archive/postings.csv and Data/raw/linkedin_jobs/postings.csv are identical (same size and MD5).

## DataStax LinkedIn Job Listings

- **ID:** `datastax_linkedin`
- **Location:** `Data/raw/datastax_linkedin`
- **Source:** Hugging Face (DataStax); original copy at ./clean_jobs.csv with ./gitattributes
- **Temporal:** historical
- **Usage:** retrieval corpus, job parsing
- **Privacy:** Public job postings.

| File | Format | Size | Rows/records |
|---|---|---|---|
| `Data/raw/datastax_linkedin/postings.csv` | csv | 4.5 MB | 1048 |

**`postings.csv` columns:** `id`, `title`, `company`, `location`, `link`, `source`, `date_posted`, `work_type`, `employment_type`, `description`

> Original filename in the workspace was clean_jobs.csv; stored here as postings.csv next to its .gitattributes, separate from the Kaggle postings.csv.
> work_type and employment_type columns are entirely/mostly null — see missing statistics.

## LinkedIn Job Posts Insights (XLSX)

- **ID:** `linkedin_insights`
- **Location:** `Data/raw/linkedin_job_insights`
- **Source:** Workspace file ./linkedin_job_posts_insights.xlsx
- **Temporal:** historical
- **Usage:** analytics, cross-source dedup
- **Privacy:** Public posting metadata.

| File | Format | Size | Rows/records |
|---|---|---|---|
| `Data/raw/linkedin_job_insights/linkedin_job_posts_insights.xlsx` | xlsx | 1.9 MB | Sheet1: 31597 |

> No job description or URL column: usable for market analytics and dedup cross-reference, not for text retrieval. Excluded from the recommendation index by default.

## RemoteOK public job feed (https://remoteok.com/api)

- **ID:** `remoteok`
- **Location:** `Data/raw/remoteok`
- **Source:** Downloaded API JSON
- **Temporal:** live snapshot
- **Usage:** retrieval corpus (current jobs)
- **Privacy:** Public postings.

| File | Format | Size | Rows/records |
|---|---|---|---|
| `Data/raw/remoteok/remoteok_jobs.json` | json | 559.9 KB | 99 |

> First array element is the API legal notice, not a job.
> Some text fields are double-encoded UTF-8 (mojibake); the loader repairs them without altering raw files.
> Attribution required: link back to the RemoteOK job URL and mention Remote OK as the source.

## Adzuna Jobs API snapshots

- **ID:** `adzuna`
- **Location:** `Data/raw/adzuna`
- **Source:** https://api.adzuna.com (credentials via env)
- **Temporal:** live snapshot
- **Usage:** retrieval corpus (current jobs)
- **Privacy:** Public postings.

| File | Format | Size | Rows/records |
|---|---|---|---|
| `Data/raw/adzuna/adzuna_us_20261009T011206_1.json` | json | 73.0 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011206_2.json` | json | 70.9 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011207_1.json` | json | 70.6 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011208_2.json` | json | 70.6 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011209_1.json` | json | 74.7 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011210_2.json` | json | 75.9 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011211_1.json` | json | 71.5 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011212_2.json` | json | 72.0 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011213_1.json` | json | 71.5 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011214_2.json` | json | 72.8 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011215_1.json` | json | 73.6 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011216_2.json` | json | 73.2 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011217_1.json` | json | 72.1 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011219_2.json` | json | 71.3 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011220_1.json` | json | 73.0 KB |  |
| `Data/raw/adzuna/adzuna_us_20261009T011221_2.json` | json | 72.7 KB |  |

> Raw API responses saved verbatim per fetch.

## Kaggle Resume Dataset (snehaanbhawal/resume-dataset)

- **ID:** `kaggle_resumes`
- **Location:** `Data/raw/resume_dataset`
- **Source:** Workspace ./archive-2/
- **Temporal:** historical
- **Usage:** candidate pool for job->candidate matching, resume parsing evaluation pool, occupation-family classifier (real category labels), annotation sampling
- **Privacy:** Resumes are anonymised by the publisher (names/companies replaced), but may still contain quasi-identifiers. Treated as sensitive: never sent to hosted LLMs by default.

| File | Format | Size | Rows/records |
|---|---|---|---|
| `Data/raw/resume_dataset/Resume.csv` | csv | 53.7 MB | 2484 |
| `Data/raw/resume_dataset/data` | pdf | 59.1 MB | 2484 |

**`Resume.csv` columns:** `ID`, `Resume_str`, `Resume_html`, `Category`

**Relationships / FK coverage:**

```json
{
 "csv_ids_with_pdf": 2484,
 "csv_rows": 2484,
 "category_agreement_csv_vs_folder": 2484
}
```


## resume_corpus (Jiechieu & Tsopze, 2020) — multi-labelled IT resumes

- **ID:** `resume_corpus`
- **Location:** `Data/raw/resume_corpus`
- **Source:** Workspace ./resume_corpus-master/ (GitHub repository snapshot)
- **Temporal:** historical
- **Usage:** occupation-normalization evaluation (expert raw-title -> class mappings), candidate pool, classical extraction training text
- **Privacy:** Real resumes scraped from public job boards; contain locations and employer names. Treated as sensitive.

| File | Format | Size | Rows/records |
|---|---|---|---|
| `Data/raw/resume_corpus/README.md` | md | 1.1 KB |  |
| `Data/raw/resume_corpus/normlized_classes.txt` | txt | 1.2 MB |  |
| `Data/raw/resume_corpus/resume_samples.zip` | zip | 60.9 MB |  |
| `Data/raw/resume_corpus/resumes_corpus.zip` | zip | 85.2 MB |  |
| `Data/raw/resume_corpus/skills_it.txt` | txt | 14.1 MB |  |

**Structure:**

```json
{
 "resume_samples.zip": {
  "member": "resume_samples.txt",
  "records": 29783,
  "format": "id:::occupation labels (';'-separated):::resume text, records separated by '::::::' lines",
  "distinct_raw_labels": 87890,
  "top_raw_labels": {
   "Python Developer": 4244,
   "Javascript": 4006,
   "Software Developer": 3242,
   "Java Developer": 3111,
   "Web Developer": 2818,
   "Systems Administrator": 2714,
   "Git": 2629,
   "Active Directory": 2069,
   "Network Administrator": 2060,
   "System Administrator": 2046
  }
 },
 "resumes_corpus.zip": {
  "txt_files": 29783,
  "lab_files": 29783,
  "label_distribution": {
   "Software_Developer": 15013,
   "Web_Developer": 6594,
   "Systems_Administrator": 5969,
   "Project_manager": 4550,
   "Network_Administrator": 4460,
   "Front_End_Developer": 3977,
   "Database_Administrator": 3299,
   "Java_Developer": 3252,
   "Security_Analyst": 3022,
   "Python_Developer": 2836
  }
 },
 "normlized_classes.txt": {
  "lines": 19465,
  "format": "raw occupation title:normalized class[,class]",
  "normalized_classes": {
   "Project_manager": 4603,
   "Systems_Administrator": 3089,
   "Software_Developer": 2718,
   "Web_Developer": 2652,
   "Database_Administrator": 2113,
   "Network_Administrator": 2049,
   "Security_Analyst": 2046,
   "Front_End_Developer": 1632,
   "Java_Developer": 1053,
   "Python_Developer": 907
  }
 },
 "skills_it.txt": {
  "records": 2438,
  "format": "Same id:::labels:::text layout as resume_samples.txt (it is a resume subset, not a skill list, despite the filename)"
 }
}
```

> The README calls the archive 'resumes_sample.zip'; the actual file is resume_samples.zip.
> The normalized classes file is spelled 'normlized_classes.txt' in the repository.

## O*NET Database O*NET 31.0 Database August 2026 Release -------------------

- **ID:** `onet`
- **Location:** `Data/raw/onet`
- **Source:** Workspace ./db_31_0_csv/ (onetcenter.org, CC BY 4.0)
- **Temporal:** reference taxonomy (2026 release)
- **Usage:** primary skill taxonomy, occupation normalization, transferable skills, occupation relations
- **Privacy:** Public reference data.

| File | Format | Size | Rows/records |
|---|---|---|---|
| `Data/raw/onet/abilities.csv` | csv | 12.1 MB | 94640 |
| `Data/raw/onet/abilities_to_work_activities.csv` | csv | 30.7 KB | 381 |
| `Data/raw/onet/abilities_to_work_context.csv` | csv | 11.5 KB | 139 |
| `Data/raw/onet/career_interest_type_keywords.csv` | csv | 2.6 KB | 75 |
| `Data/raw/onet/career_interest_types.csv` | csv | 1.1 MB | 8307 |
| `Data/raw/onet/content_model_reference.csv` | csv | 230.3 KB | 3006 |
| `Data/raw/onet/education.csv` | csv | 1.9 MB | 11495 |
| `Data/raw/onet/education_categories.csv` | csv | 3.1 KB | 24 |
| `Data/raw/onet/emerging_tasks.csv` | csv | 53.1 KB | 338 |
| `Data/raw/onet/essential_skills.csv` | csv | 2.2 MB | 18200 |
| `Data/raw/onet/essential_skills_to_work_activities.csv` | csv | 8.3 KB | 110 |
| `Data/raw/onet/essential_skills_to_work_context.csv` | csv | 2.8 KB | 39 |
| `Data/raw/onet/gwas_to_iwas.csv` | csv | 34.2 KB | 332 |
| `Data/raw/onet/gwas_to_iwas_to_dwas.csv` | csv | 343.5 KB | 2087 |
| `Data/raw/onet/interests_illustrative_activities.csv` | csv | 14.6 KB | 188 |
| `Data/raw/onet/interests_illustrative_occupations.csv` | csv | 14.7 KB | 186 |
| `Data/raw/onet/job_titles.csv` | csv | 4.3 MB | 54269 |
| `Data/raw/onet/job_zone_reference.csv` | csv | 3.6 KB | 4 |
| `Data/raw/onet/job_zones.csv` | csv | 58.5 KB | 923 |
| `Data/raw/onet/knowledge.csv` | csv | 7.6 MB | 60060 |
| `Data/raw/onet/level_scale_anchors.csv` | csv | 44.6 KB | 483 |
| `Data/raw/onet/occupation_data.csv` | csv | 261.7 KB | 1016 |
| `Data/raw/onet/occupation_level_metadata.csv` | csv | 3.6 MB | 32280 |
| `Data/raw/onet/related_occupations.csv` | csv | 2.0 MB | 18460 |
| `Data/raw/onet/sample_of_reported_titles.csv` | csv | 587.0 KB | 8189 |
| `Data/raw/onet/scales_reference.csv` | csv | 1006.0 B | 33 |
| `Data/raw/onet/software_skills.csv` | csv | 3.4 MB | 31821 |
| `Data/raw/onet/specific_interest_areas.csv` | csv | 8.3 MB | 73062 |
| `Data/raw/onet/specific_interest_areas_to_career_interest_types.csv` | csv | 2.4 KB | 53 |
| `Data/raw/onet/survey_booklet_locations.csv` | csv | 13.1 KB | 211 |
| `Data/raw/onet/task_categories.csv` | csv | 446.0 B | 7 |
| `Data/raw/onet/task_ratings.csv` | csv | 37.2 MB | 165780 |
| `Data/raw/onet/task_statements.csv` | csv | 3.3 MB | 18838 |
| `Data/raw/onet/tasks_to_dwas.csv` | csv | 5.3 MB | 24087 |
| `Data/raw/onet/training_and_experience.csv` | csv | 4.3 MB | 26812 |
| `Data/raw/onet/training_and_experience_categories.csv` | csv | 3.4 KB | 29 |
| `Data/raw/onet/transferable_skills.csv` | csv | 5.8 MB | 45500 |
| `Data/raw/onet/transferable_skills_to_work_activities.csv` | csv | 9.8 KB | 122 |
| `Data/raw/onet/transferable_skills_to_work_context.csv` | csv | 4.6 KB | 57 |
| `Data/raw/onet/work_activities.csv` | csv | 11.3 MB | 74702 |
| `Data/raw/onet/work_context.csv` | csv | 50.5 MB | 305389 |
| `Data/raw/onet/work_context_categories.csv` | csv | 29.7 KB | 281 |
| `Data/raw/onet/work_styles.csv` | csv | 4.1 MB | 37422 |
| `Data/raw/onet/work_styles_to_work_activities.csv` | csv | 22.4 KB | 303 |
| `Data/raw/onet/work_styles_to_work_context.csv` | csv | 19.3 KB | 266 |

**`abilities.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Not Relevant`, `Date`, `Domain Source`

**`abilities_to_work_activities.csv` columns:** `Abilities Element ID`, `Abilities Element Name`, `Work Activities Element ID`, `Work Activities Element Name`

**`abilities_to_work_context.csv` columns:** `Abilities Element ID`, `Abilities Element Name`, `Work Context Element ID`, `Work Context Element Name`

**`career_interest_type_keywords.csv` columns:** `Element ID`, `Element Name`, `Keyword`, `Keyword Type`

**`career_interest_types.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `Date`, `Domain Source`

**`content_model_reference.csv` columns:** `Element ID`, `Element Name`, `Description`

**`education.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Category`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Date`, `Domain Source`

**`education_categories.csv` columns:** `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Category`, `Category Description`

**`emerging_tasks.csv` columns:** `O*NET-SOC Code`, `Title`, `Task`, `Category`, `Original Task ID`, `Original Task`, `Date`, `Domain Source`

**`essential_skills.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Not Relevant`, `Date`, `Domain Source`

**`essential_skills_to_work_activities.csv` columns:** `Essential Skills Element ID`, `Essential Skills Element Name`, `Work Activities Element ID`, `Work Activities Element Name`

**`essential_skills_to_work_context.csv` columns:** `Essential Skills Element ID`, `Essential Skills Element Name`, `Work Context Element ID`, `Work Context Element Name`

**`gwas_to_iwas.csv` columns:** `GWA Element ID`, `GWA Element Name`, `IWA Element ID`, `IWA Element Name`

**`gwas_to_iwas_to_dwas.csv` columns:** `GWA Element ID`, `GWA Element Name`, `IWA Element ID`, `IWA Element Name`, `DWA Element ID`, `DWA Element Name`

**`interests_illustrative_activities.csv` columns:** `Element ID`, `Element Name`, `Interest Type`, `Activity`

**`interests_illustrative_occupations.csv` columns:** `Element ID`, `Element Name`, `Interest Type`, `O*NET-SOC Code`, `Title`

**`job_titles.csv` columns:** `O*NET-SOC Code`, `Title`, `Job Title`, `Short Title`, `Source(s)`

**`job_zone_reference.csv` columns:** `Job Zone`, `Name`, `Experience`, `Education`, `Job Training`, `Examples`, `SVP Range`

**`job_zones.csv` columns:** `O*NET-SOC Code`, `Title`, `Job Zone`, `Date`, `Domain Source`

**`knowledge.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Not Relevant`, `Date`, `Domain Source`

**`level_scale_anchors.csv` columns:** `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Anchor Value`, `Anchor Description`

**`occupation_data.csv` columns:** `O*NET-SOC Code`, `Title`, `Description`

**`occupation_level_metadata.csv` columns:** `O*NET-SOC Code`, `Title`, `Item`, `Response`, `N`, `Percent`, `Date`

**`related_occupations.csv` columns:** `O*NET-SOC Code`, `Title`, `Related O*NET-SOC Code`, `Related Title`, `Relatedness Tier`, `Index`

**`sample_of_reported_titles.csv` columns:** `O*NET-SOC Code`, `Title`, `Reported Job Title`, `Shown in My Next Move`

**`scales_reference.csv` columns:** `Scale ID`, `Scale Name`, `Minimum`, `Maximum`

**`software_skills.csv` columns:** `O*NET-SOC Code`, `Title`, `Workplace Example`, `Element ID`, `Element Name`, `Hot Technology`, `In Demand`

**`specific_interest_areas.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `Date`, `Domain Source`

**`specific_interest_areas_to_career_interest_types.csv` columns:** `Specific Interest Areas Element ID`, `Specific Interest Areas Element Name`, `Career Interest Types Element ID`, `Career Interest Types Element Name`

**`survey_booklet_locations.csv` columns:** `Element ID`, `Element Name`, `Survey Item Number`, `Scale ID`, `Scale Name`

**`task_categories.csv` columns:** `Scale ID`, `Scale Name`, `Category`, `Category Description`

**`task_ratings.csv` columns:** `O*NET-SOC Code`, `Title`, `Task ID`, `Task`, `Scale ID`, `Scale Name`, `Category`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Date`, `Domain Source`

**`task_statements.csv` columns:** `O*NET-SOC Code`, `Title`, `Task ID`, `Task`, `Task Type`, `Incumbents Responding`, `Date`, `Domain Source`

**`tasks_to_dwas.csv` columns:** `O*NET-SOC Code`, `Title`, `Task ID`, `Task`, `DWA Element ID`, `DWA Element Name`, `Date`, `Domain Source`

**`training_and_experience.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Category`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Date`, `Domain Source`

**`training_and_experience_categories.csv` columns:** `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Category`, `Category Description`

**`transferable_skills.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Not Relevant`, `Date`, `Domain Source`

**`transferable_skills_to_work_activities.csv` columns:** `Transferable Skills Element ID`, `Transferable Skills Element Name`, `Work Activities Element ID`, `Work Activities Element Name`

**`transferable_skills_to_work_context.csv` columns:** `Transferable Skills Element ID`, `Transferable Skills Element Name`, `Work Context Element ID`, `Work Context Element Name`

**`work_activities.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Not Relevant`, `Date`, `Domain Source`

**`work_context.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Category`, `Data Value`, `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Not Relevant`, `Date`, `Domain Source`

**`work_context_categories.csv` columns:** `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Category`, `Category Description`

**`work_styles.csv` columns:** `O*NET-SOC Code`, `Title`, `Element ID`, `Element Name`, `Scale ID`, `Scale Name`, `Data Value`, `Date`, `Domain Source`

**`work_styles_to_work_activities.csv` columns:** `Work Styles Element ID`, `Work Styles Element Name`, `Work Activities Element ID`, `Work Activities Element Name`

**`work_styles_to_work_context.csv` columns:** `Work Styles Element ID`, `Work Styles Element Name`, `Work Context Element ID`, `Work Context Element Name`

**Relationships / FK coverage:**

```json
{
 "*.O*NET-SOC Code -> occupation_data.O*NET-SOC Code": "all occupation-level files",
 "*.Element ID -> content_model_reference.Element ID": "skills/knowledge/abilities/activities",
 "related_occupations.Related O*NET-SOC Code -> occupation_data": "occupation graph"
}
```

> Release 31.0 uses new file names: essential_skills.csv and transferable_skills.csv (the former 'Skills' file split), software_skills.csv (former Technology Skills).
> software_skills Element IDs identify technology categories; individual tools (Workplace Example) have no O*NET ID, so the project derives stable IDs 'onet-tech:<slug>'.

## Open Skills Management Tool (OSMT) — import resources

- **ID:** `osmt`
- **Location:** `Data/raw/osmt`
- **Source:** Workspace ./osmt-develop/ (WGU OSMT source repository). Only import/ CSVs, README and LICENSE were copied; the repository (incl. 1.2 GB UI sources) is preserved in place.
- **Temporal:** reference
- **Usage:** SOC hierarchy names, skill schema reference
- **Privacy:** Public.

| File | Format | Size | Rows/records |
|---|---|---|---|
| `Data/raw/osmt/import/BLS-Import.csv` | csv | 290.8 KB | 1447 |
| `Data/raw/osmt/import/Skills-Import.csv` | csv | 936.0 B | 2 |
| `Data/raw/osmt/import/oNet-Import.csv` | csv | 260.8 KB | 1016 |
| `Data/raw/osmt/import/osmt-batch-import-example.csv` | csv | 333.0 B | 0 |

**`BLS-Import.csv` columns:** `SOC Group`, `SOC Code`, `SOC Title`, `SOC Definition`

**`Skills-Import.csv` columns:** `Collection`, `Skill Name`, `Skill Category`, `Contextualized Skill Statement`, `Keywords`, `BLS Major Group`, `BLS Minor Group`, `BLS Broad Occupation`, `BLS Detailed Occupation`, `O*NET Job Role`, `Professional Standards`, `Certifications`, `Alignment Title`, `Alignment`

**`oNet-Import.csv` columns:** `O*NET-SOC Code`, `Title`, `Description`

**`osmt-batch-import-example.csv` columns:** `RSD Name`, `Authors`, `Skill Statement`, `Categories`, `Keywords`, `Standards`, `Certifications`, `Occupation Major Groups`, `Occupation Minor Groups`, `Broad Occupations`, `Detailed Occupations`, `O*NET Job Codes`, `Employers`, `Alignment Name`, `Alignment URL`, `Alignment Framework`, `Alignment 2 Name`, `Alignment 2 URL`, `Alignment 2 Framework`

> OSMT is application source code (Kotlin/Angular), not a dataset. Its import/ folder contains a BLS SOC hierarchy (BLS-Import.csv), an O*NET occupation list identical in structure to occupation_data.csv, a 2-row example Rich Skill Descriptor file, and a batch-import template.
> Used: BLS-Import.csv for SOC major/minor group names; RSD columns inform the Skill node schema.

## Cross-source overlap

```json
{
 "title_company_key": {
  "linkedin_kaggle \u2229 datastax_linkedin": 14,
  "linkedin_kaggle \u2229 linkedin_insights": 58,
  "linkedin_kaggle \u2229 remoteok": 0,
  "linkedin_kaggle \u2229 adzuna": 2,
  "datastax_linkedin \u2229 linkedin_insights": 5,
  "datastax_linkedin \u2229 remoteok": 0,
  "datastax_linkedin \u2229 adzuna": 0,
  "linkedin_insights \u2229 remoteok": 0,
  "linkedin_insights \u2229 adzuna": 2,
  "remoteok \u2229 adzuna": 0
 },
 "description_hash": {
  "linkedin_kaggle \u2229 datastax_linkedin": 0,
  "linkedin_kaggle \u2229 remoteok": 0,
  "datastax_linkedin \u2229 remoteok": 0
 },
 "sizes": {
  "linkedin_kaggle": {
   "distinct_title_company": 96400,
   "distinct_description_hash": 107159
  },
  "datastax_linkedin": {
   "distinct_title_company": 814,
   "distinct_description_hash": 931
  },
  "linkedin_insights": {
   "distinct_title_company": 10445,
   "distinct_description_hash": 0
  },
  "remoteok": {
   "distinct_title_company": 99,
   "distinct_description_hash": 99
  },
  "adzuna": {
   "distinct_title_company": 394,
   "distinct_description_hash": 0
  }
 }
}
```
"""Resume/job parsing, text extraction, skill & occupation normalisation."""
import io

import pytest

from common.schemas import Job
from normalization.aliases import ALIASES
from normalization.skill_normalizer import get_skill_normalizer
from normalization.taxonomy import load_taxonomy
from parsing.classical_parser import ClassicalJobParser, ClassicalResumeParser, candidate_name, months_between
from parsing.sections import JOB_HEADERS, RESUME_HEADERS, segment
from parsing.text_extraction import UnsupportedFile, extract_text


@pytest.fixture(scope="module")
def parsed(resume_text):
    return ClassicalResumeParser().parse(resume_text, "cand-test", "ds_resume.txt")


# -------------------------------------------------------------- normalisation
def test_every_curated_alias_resolves():
    tax = load_taxonomy()
    names = {n.lower() for n in tax.skills["name"]}
    assert all(t.lower() in names for t in ALIASES.values())


def test_related_skills_are_not_identical():
    n = get_skill_normalizer()
    assert n.normalize("pytorch").canonical_name == "PyTorch"
    assert n.normalize("tensorflow").canonical_name == "TensorFlow"
    assert n.normalize("pytorch").canonical_id != n.normalize("tensorflow").canonical_id
    assert n.normalize("torch").canonical_id is None              # abstain rather than guess
    assert n.normalize("Java").canonical_id != n.normalize("JavaScript").canonical_id


def test_related_skill_pairs_are_listed_once():
    tax = load_taxonomy()
    for sid, rel in tax.related_skills.items():
        others = [o for o, _ in rel]
        assert len(others) == len(set(others)), sid


def test_normalisation_methods():
    n = get_skill_normalizer()
    assert n.normalize("k8s").method == "alias"
    assert n.normalize("machine learning").method == "exact"
    r = n.normalize("Tensorflow 2", strategy="classical")
    assert r.canonical_name == "TensorFlow" and r.method == "tfidf_char"
    assert n.normalize("completely unknown thing").method == "unresolved"


def test_gazetteer_case_and_context_rules():
    n = get_skill_normalizer()
    found = {m.mention: m.canonical_name for m in n.extract(
        "Skills: Python, R, Go, SQL\nI excel at teamwork. Let's go. ADA compliance training.", "d")}
    assert found.get("R") == "R" and found.get("Go") == "Go" and "SQL" in found
    assert "excel" not in found and "go" not in found and "ADA" not in found


def test_occupation_normaliser():
    from normalization.occupation_normalizer import get_occupation_normalizer

    n = get_occupation_normalizer()
    assert n.normalize("Senior Data Scientist").canonical_title == "Data Scientists"
    assert n.normalize("Full Stack Developer").code.startswith("15-125")   # software/web dev, not photography
    assert n.normalize("Registered Nurse (Night Shift)").code == "29-1141.00"


# -------------------------------------------------------------- parsing
def test_sections():
    secs = segment("Summary   text   Experience   Engineer 2019 to 2020   Education   BS", RESUME_HEADERS)
    assert [s.name for s in secs] == ["summary", "experience", "education"]
    jsecs = segment("About us\nWe\nRequirements:\n5 years Python\nNice to have:\nGo", JOB_HEADERS, "intro")
    assert [s.name for s in jsecs] == ["about", "requirements", "preferred"]


def test_resume_parse_with_provenance(parsed, resume_text):
    names = {s.canonical_name for s in parsed.skills}
    assert {"Python", "PyTorch", "Docker", "Tableau", "Scikit-learn"} <= names
    for ev in parsed.all_evidence():   # every fact points to an exact span
        assert resume_text[ev.start_offset:ev.end_offset] == ev.source_text
        assert ev.extractor and ev.extractor_version and 0 <= ev.confidence <= 1
    titles = [e.title for e in parsed.experiences]
    assert "Senior Data Scientist" in titles and "Data Analyst" in titles
    assert parsed.total_experience_years and parsed.total_experience_years > 4
    assert {e.level for e in parsed.education} == {"master", "bachelor"}
    assert any("AWS Certified" in c.name for c in parsed.credentials)
    assert any(o.canonical_title == "Data Scientists" for o in parsed.occupations)
    assert "jane.tester@example.com" not in {s.mention for s in parsed.skills}


def test_job_parse_required_preferred_years():
    text = ("About us\nWe build software.\nRequirements:\n- 3+ years of experience with Python and SQL\n"
            "- Bachelor's degree in Computer Science\nNice to have:\n- Experience with Kubernetes\n")
    job = ClassicalJobParser().parse(Job(job_id="t:1", source_job_id="1", source="t", title="Data Engineer",
                                         description=text))
    imp = {r.skill.canonical_name: r.importance for r in job.requirements if r.skill}
    assert imp["Python"] == "required" and imp["Kubernetes"] == "preferred"
    assert any(r.kind == "experience" and r.years == 3 for r in job.requirements)
    assert any(r.kind == "education" and r.education_level == "bachelor" for r in job.requirements)
    for r in job.requirements:
        for ev in r.evidence:
            assert text[ev.start_offset:ev.end_offset] == ev.source_text


def test_months_between():
    assert months_between("Jan 2021", "Dec 2021") == 12
    assert months_between("06/2018", "12/2020") == 31


# -------------------------------------------------------------- file extraction
def test_txt_docx_pdf_and_rejection():
    import docx

    assert extract_text(b"Hello resume\nPython").text.startswith("Hello")
    d = docx.Document()
    d.add_paragraph("Skills: Python, SQL")
    buf = io.BytesIO()
    d.save(buf)
    assert "Python" in extract_text(buf.getvalue(), "x.docx").text
    from common.config import get_settings

    pdf = next((get_settings().path("raw") / "resume_dataset" / "data" / "HR").glob("*.pdf")).read_bytes()
    out = extract_text(pdf, "x.pdf")
    assert out.file_type == "pdf" and len(out.text) > 200
    with pytest.raises(UnsupportedFile):
        extract_text(b"MZ\x90\x00\x03\x00binary")
    with pytest.raises(UnsupportedFile):
        extract_text(b"PK\x03\x04notadocx")


def test_fragmented_pdf_text_is_detected():
    from parsing.text_extraction import _fragmented

    words = "Data Engineer with five years of experience building pipelines in Python and SQL".split()
    assert _fragmented("\n \n".join(words * 4))  # pypdf output for Google Docs PDFs: one word per line
    assert not _fragmented("\n".join([" ".join(words)] * 4))


PIPE_RESUME = """JANE DOE
(555) 010-0000 | jane@example.com
SUMMARY
Data Engineer with 5+ years building batch and streaming pipelines.
PROFESSIONAL EXPERIENCE
Senior Data Engineer, Platform Team | Acme Analytics | Denver, CO | Mar 2022 - Present
• Built Spark pipelines in Python processing 2 TB per day
Analyst | Globex Corp | Austin, TX | Jan 2019 - Feb 2022
• Wrote SQL reports for finance stakeholders
PROJECTS
Personal Project | Streaming Fraud Detector: Built a Kafka consumer that scores card transactions and writes
alerts to PostgreSQL with sub-second latency for downstream review
Demo: https://example.com/demo
Capstone Project | Housing Price Model: Trained gradient-boosted models on county records
Other Projects GitHub: github.com/example
EDUCATION
Master of Science in Data Science | GPA: 3.9
Example State University | May 2018
Bachelor of Engineering
Sample Institute of Technology
CERTIFICATIONS
AWS Certified Data Engineer | Databricks Data Engineer Associate | Example Cloud Fundamentals
"""


def test_pipe_delimited_resume_layout():
    c = ClassicalResumeParser().parse(PIPE_RESUME, "cand-pipe", "pipe.txt")
    roles = [(e.title, e.company) for e in c.experiences]
    assert roles == [("Senior Data Engineer, Platform Team", "Acme Analytics"), ("Analyst", "Globex Corp")]
    assert [e.institution for e in c.education] == ["Example State University", "Sample Institute of Technology"]
    assert [x.name for x in c.credentials] == ["AWS Certified Data Engineer", "Databricks Data Engineer Associate",
                                               "Example Cloud Fundamentals"]
    assert [p.name for p in c.projects] == ["Streaming Fraud Detector", "Housing Price Model"]
    assert "PostgreSQL" in c.projects[0].description and "example.com/demo" in c.projects[0].description
    # the comma-qualified title resolves through its head, with evidence on just the head
    occ = c.experiences[0].occupation
    assert occ is not None and occ.evidence[0].source_text == "Senior Data Engineer"
    # …but a one-word head is too generic to stand in for 'Analyst, <practice>'
    c2 = ClassicalResumeParser().parse(PIPE_RESUME.replace("Analyst | Globex", "Analyst, AI and Data Practice | Globex"),
                                       "cand-pipe2", "pipe.txt")
    occ2 = c2.experiences[1].occupation
    assert c2.experiences[1].title == "Analyst, AI and Data Practice"
    assert occ2 is None or occ2.mention != "Analyst"
    for e in c.experiences:
        ev = e.evidence[0]
        assert PIPE_RESUME[ev.start_offset:ev.end_offset] == ev.source_text


STACKED_PROJECTS = """Jordan Example
jordan@example.com
PROJECTS
Distributed Traffic Management and Prediction System
Aug 2025 – Dec 2025
Python, Apache Kafka, MongoDB, Docker, LightGBM, Streamlit
 •
Engineered a real-time streaming pipeline ingesting sensor data via Apache Kafka into a 3-node MongoDB replica set,
sustaining continuous high-volume ingestion.
•
Executed controlled failover tests, achieving automatic primary re-election in 2.3 seconds.
Lights! Camera! Impact! — Movie Data Analytics
Jan 2025 – May 2025
Python, Pandas, VADER, D3.js
 •
Built data-integration pipelines merging 10+ heterogeneous sources into a unified analytics dataset.
"""


def test_stacked_title_date_stack_projects():
    """Title / date line / tech-stack line / bullets: one project each, never one per line."""
    c = ClassicalResumeParser().parse(STACKED_PROJECTS, "cand-stack", "stack.txt")
    assert [p.name for p in c.projects] == ["Distributed Traffic Management and Prediction System",
                                            "Lights! Camera! Impact! — Movie Data Analytics"]
    assert c.projects[0].description.startswith("Engineered a real-time")
    # the stack line is inside the project's span, so its skills are linked to the project
    assert {"Apache Kafka", "Docker"} <= {s.mention for s in c.projects[0].skills}
    # same layout with the dates tab-separated on the title line (DOCX export)
    tabbed = STACKED_PROJECTS.replace("System\nAug 2025", "System\tAug 2025")
    c2 = ClassicalResumeParser().parse(tabbed, "cand-stack2", "stack.txt")
    assert c2.projects[0].name == "Distributed Traffic Management and Prediction System"
    assert len(c2.projects) == 2


@pytest.mark.parametrize("text,expected", [
    ("KRISH NAIK\n602-555-0100 | krish@example.com", "Krish Naik"),
    ("Mary-Jane O'Brien | Austin, TX\n(555) 123-4567", "Mary-Jane O'Brien"),
    ("J. R. Smith\njr@example.org", "J. R. Smith"),
    ("Software Engineer\njohn@example.com", None),                    # a role, not a name
    ("BPO TEAM LEADER       Summary    Seeking to provide support", None),   # anonymised corpus resume
    ("John Smith\nSummary\nNo contact details in the header", None),  # no contact signal: abstain
    ("", None),
])
def test_candidate_name(text, expected):
    assert candidate_name(text) == expected


def _edu(text):
    c = ClassicalResumeParser().parse(text, "cand-edu", "edu.txt")
    return [(e.degree, e.field, e.institution, e.year) for e in c.education]


@pytest.mark.parametrize("text,expected", [
    # School (dates) / Degree blocks — each degree keeps its own school and its own (end) year
    ("EDUCATION\nExample State University Aug 2024 – Dec 2026\nM.S. Computer Science | GPA: 3.6\n"
     "Sample Institute of Technology Jul 2020 – May 2024\nB.E. Computer Engineering\nSKILLS\nPython\n",
     [("M.S.", "Computer Science", "Example State University", "2026"),
      ("B.E.", "Computer Engineering", "Sample Institute of Technology", "2024")]),
    # Degree / School (date) blocks — a date on one entry's line never leaks into the next entry
    ("EDUCATION\nMaster of Science in Data Science | GPA: 3.9\nExample State University | May 2025\n"
     "Bachelor of Engineering\nSample Institute of Technology\nSKILLS\nPython\n",
     [("Master of Science", "Data Science", "Example State University", "2025"),
      ("Bachelor of Engineering", "Engineering", "Sample Institute of Technology", None)]),
    # flattened 'Degree : School - Field' with the abbreviation restated in parentheses
    ("Education   Bachelor of Laws (LLB)  :  Example Law University  -  Contract Law   City  ,   State   "
     "Associate of Arts  :  Sample Valley College  -  Interior Design   City  ,   State",
     [("Bachelor of Laws", "Contract Law", "Example Law University", None),
      ("Associate of Arts", "Interior Design", "Sample Valley College", None)]),
])
def test_education_layouts(text, expected):
    assert _edu(text) == expected


def test_education_rejects_bare_words():
    # 'Master' inside a school name is not a degree; a bare 'College' is not an institution
    assert _edu("EDUCATION\nRoad Master Truck Driving School 2000\nClass A CDL\nSKILLS\nDriving\n") == []
    out = _edu("EDUCATION\nHigh School Diploma , College Prep studies   1998   Example Senior High\nSKILLS\nX\n")
    assert out == [("High School Diploma", "College Prep", "Example Senior High", "1998")]

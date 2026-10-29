"""Local skill supplement and curated aliases.

O*NET is the primary taxonomy. It names tools precisely (``PyTorch``,
``TensorFlow``, ``Docker``) but has no entries for many cross-cutting
concepts (``machine learning``, ``agile``). Those are added here under the
explicit ``local`` taxonomy namespace so they are never confused with O*NET
entries. Every alias target must resolve to an existing canonical name —
``tests/unit/test_normalization.py`` enforces this.

Aliases map surface forms to *the same* concept only. Related-but-distinct
skills (PyTorch vs TensorFlow, Java vs JavaScript) are never aliased.
"""
from __future__ import annotations

import re

# name -> category
LOCAL_SKILLS: dict[str, str] = {
    # data / AI
    "Machine learning": "data science", "Deep learning": "data science",
    "Natural language processing": "data science", "Computer vision": "data science",
    "Data analysis": "data science", "Data science": "data science", "Statistics": "data science",
    "Data visualization": "data science", "Big data": "data science", "Data engineering": "data science",
    "Extract, transform, load ETL": "data science", "Data mining": "data science",
    "Data modeling": "data science", "Feature engineering": "data science",
    "Time series analysis": "data science", "Regression analysis": "data science",
    "A/B testing": "data science", "Reinforcement learning": "data science",
    "Recommender systems": "data science", "Information retrieval": "data science",
    "Knowledge graphs": "data science", "Large language models": "data science",
    "Generative AI": "data science", "Prompt engineering": "data science",
    "Retrieval-augmented generation": "data science", "MLOps": "data science",
    "scikit-learn": "data science", "Predictive modeling": "data science",
    "Business intelligence": "data science", "Data warehousing": "data science",
    "LightGBM": "data science", "Matplotlib": "data science", "OpenCV": "data science",
    "Streamlit": "software engineering", "Next.js": "software engineering", "FastAPI": "software engineering",
    "Google Colab": "data science", "D3.js": "data science",
    # software engineering
    "Software development": "software engineering", "Web development": "software engineering",
    "Front-end development": "software engineering", "Back-end development": "software engineering",
    "Full-stack development": "software engineering", "Mobile development": "software engineering",
    "Object-oriented programming": "software engineering", "REST APIs": "software engineering",
    "Microservices": "software engineering", "Unit testing": "software engineering",
    "Test automation": "software engineering", "Continuous integration and delivery CI/CD": "software engineering",
    "DevOps": "software engineering", "Cloud computing": "software engineering",
    "Version control": "software engineering", "System design": "software engineering",
    "Data structures and algorithms": "software engineering", "Debugging": "software engineering",
    "Software architecture": "software engineering", "Database administration": "software engineering",
    "Network administration": "it", "Systems administration": "it", "Technical support": "it",
    "Cybersecurity": "it", "Penetration testing": "it", "Network security": "it",
    "Troubleshooting": "it", "Help desk support": "it",
    # business
    "Project management": "business", "Product management": "business", "Agile methodology": "business",
    "Scrum": "business", "Stakeholder management": "business", "Business analysis": "business",
    "Requirements gathering": "business", "Process improvement": "business", "Strategic planning": "business",
    "Budgeting": "business", "Forecasting": "business", "Financial analysis": "finance",
    "Financial reporting": "finance", "Accounting": "finance", "Bookkeeping": "finance", "Auditing": "finance",
    "Accounts payable": "finance", "Accounts receivable": "finance", "Payroll": "finance", "Tax preparation": "finance",
    "Risk management": "finance", "Underwriting": "finance", "Financial modeling": "finance",
    "Sales": "sales & marketing", "Business development": "sales & marketing", "Account management": "sales & marketing",
    "Lead generation": "sales & marketing", "Digital marketing": "sales & marketing",
    "Search engine optimization SEO": "sales & marketing", "Social media marketing": "sales & marketing",
    "Content marketing": "sales & marketing", "Email marketing": "sales & marketing", "Market research": "sales & marketing",
    "Brand management": "sales & marketing", "Public relations": "sales & marketing", "Copywriting": "sales & marketing",
    "Customer relationship management": "sales & marketing", "Event planning": "sales & marketing",
    "Recruiting": "hr", "Talent acquisition": "hr", "Onboarding": "hr", "Employee relations": "hr",
    "Performance management": "hr", "Compensation and benefits": "hr", "Training and development": "hr",
    "Supply chain management": "operations", "Logistics": "operations", "Inventory management": "operations",
    "Procurement": "operations", "Quality assurance": "operations", "Quality control": "operations",
    "Lean manufacturing": "operations", "Six Sigma": "operations", "Vendor management": "operations",
    "Operations management": "operations", "Warehouse operations": "operations",
    # other occupational
    "Customer service": "service", "Patient care": "healthcare", "Medical terminology": "healthcare",
    "Electronic health records": "healthcare", "Phlebotomy": "healthcare", "Medication administration": "healthcare",
    "Teaching": "education", "Curriculum development": "education", "Lesson planning": "education",
    "Classroom management": "education", "Tutoring": "education",
    "Food safety": "hospitality", "Food preparation": "hospitality", "Menu planning": "hospitality",
    "Graphic design": "design", "User experience design UX": "design", "User interface design UI": "design",
    "Wireframing": "design", "Prototyping": "design", "Video editing": "design", "Photography": "design",
    "Construction management": "construction", "Blueprint reading": "construction", "Estimating": "construction",
    "Legal research": "legal", "Contract management": "legal", "Litigation": "legal", "Compliance": "legal",
    "Leadership": "soft skill", "Team leadership": "soft skill", "Communication": "soft skill",
    "Teamwork": "soft skill", "Problem solving": "soft skill", "Time management": "soft skill",
    "Public speaking": "soft skill", "Presentation skills": "soft skill", "Mentoring": "soft skill",
    "Research": "soft skill", "Attention to detail": "soft skill",
}

# surface form (case-insensitive unless listed in CASE_SENSITIVE) -> canonical name
ALIASES: dict[str, str] = {
    "python3": "Python", "python 3": "Python",
    # 'Java' is also a dictionary word, so the vendor-stripped derived alias of 'Oracle Java' is never spotted;
    # curated + case-sensitive ('Java', not 'java' the coffee)
    "java": "Oracle Java", "nextjs": "Next.js", "next js": "Next.js", "colab": "Google Colab", "d3": "D3.js",
    "js": "JavaScript", "es6": "JavaScript", "ecmascript": "JavaScript",
    "ts": "TypeScript",
    "node": "Node.js", "nodejs": "Node.js", "node js": "Node.js",
    "reactjs": "React", "react.js": "React", "react js": "React",
    "vue": "Vue.js", "vuejs": "Vue.js", "angular": "Google Angular", "angularjs": "Google Angular",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "mysql": "MySQL",
    "sql": "Structured query language SQL", "t-sql": "Microsoft SQL Server", "ms sql": "Microsoft SQL Server",
    "sql server": "Microsoft SQL Server", "mssql": "Microsoft SQL Server",
    "aws": "Amazon Web Services AWS software", "amazon web services": "Amazon Web Services AWS software",
    "azure": "Microsoft Azure software", "microsoft azure": "Microsoft Azure software",
    "gcp": "Google Cloud software", "google cloud platform": "Google Cloud software", "google cloud": "Google Cloud software",
    "k8s": "Kubernetes",
    "golang": "Go",
    "html": "Hypertext markup language HTML", "html5": "Hypertext markup language HTML",
    "css": "Cascading style sheets CSS", "css3": "Cascading style sheets CSS",
    "excel": "Microsoft Excel", "ms excel": "Microsoft Excel", "advanced excel": "Microsoft Excel",
    "ms word": "Microsoft Word", "powerpoint": "Microsoft PowerPoint", "ms powerpoint": "Microsoft PowerPoint",
    "outlook": "Microsoft Outlook", "ms office": "Microsoft Office software", "microsoft office": "Microsoft Office software",
    "ms office suite": "Microsoft Office software", "power bi": "Microsoft Power BI", "powerbi": "Microsoft Power BI",
    "matlab": "The MathWorks MATLAB", "spss": "IBM SPSS Statistics", "stata": "StataCorp Stata",
    "jira": "Atlassian JIRA", "confluence": "Atlassian Confluence", "jenkins": "Jenkins CI",
    "quickbooks": "Intuit QuickBooks", "autocad": "Autodesk AutoCAD", "photoshop": "Adobe Photoshop",
    "illustrator": "Adobe Illustrator", "indesign": "Adobe InDesign",
    "salesforce": "Salesforce software", "sap": "SAP software", "workday": "Workday software",
    "hadoop": "Apache Hadoop", "spark": "Apache Spark", "apache spark": "Apache Spark", "pyspark": "PySpark",
    "kafka": "Apache Kafka", "airflow": "Apache Airflow", "ansible": "Ansible software",
    "dotnet": "Microsoft .NET Framework", "asp.net": "Microsoft ASP.NET",
    "active directory": "Microsoft Active Directory", "splunk": "Splunk Enterprise",
    "looker": "Google Looker Analytics", "android": "Google Android", "ios": "Apple iOS",
    "chatgpt": "OpenAI ChatGPT", "huggingface": "Hugging Face", "hugging face transformers": "Hugging Face",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "ml": "Machine learning", "machine-learning": "Machine learning",
    "dl": "Deep learning", "nlp": "Natural language processing", "llm": "Large language models",
    "llms": "Large language models", "genai": "Generative AI", "gen ai": "Generative AI",
    "rag": "Retrieval-augmented generation", "etl": "Extract, transform, load ETL",
    "ci/cd": "Continuous integration and delivery CI/CD", "cicd": "Continuous integration and delivery CI/CD",
    "continuous integration": "Continuous integration and delivery CI/CD",
    "rest": "REST APIs", "restful": "REST APIs", "rest api": "REST APIs", "restful apis": "REST APIs",
    "oop": "Object-oriented programming", "object oriented programming": "Object-oriented programming",
    "agile": "Agile methodology", "agile methodologies": "Agile methodology",
    "seo": "Search engine optimization SEO", "ux": "User experience design UX", "ui": "User interface design UI",
    "ux design": "User experience design UX", "ui design": "User interface design UI",
    "ui/ux": "User experience design UX", "crm": "Customer relationship management",
    "a/b tests": "A/B testing", "ab testing": "A/B testing", "bi": "Business intelligence",
    "data visualisation": "Data visualization", "visualization": "Data visualization",
    "statistical analysis": "Statistics", "data analytics": "Data analysis", "analytics": "Data analysis",
    "ehr": "Electronic health records", "emr": "Electronic health records",
    "qa": "Quality assurance", "qc": "Quality control", "sdlc": "Software development",
    "unix shell": "UNIX Shell", "shell scripting": "Shell script", "bash scripting": "Bash",
    "negotiation": "Negotiation", "persuasion": "Persuasion",
    "communication skills": "Communication", "ml ops": "MLOps", "project mgmt": "Project management",
    "amazon aws": "Amazon Web Services AWS software", "aws cloud": "Amazon Web Services AWS software",
}

# Forms that are ordinary English words (or 1–2 letters) — only accepted when
# the original casing matches and they appear in a list-like context.
CASE_SENSITIVE: set[str] = {"Java", "Go", "R", "C", "Swift", "React", "Ruby", "Spring Framework", "Flask", "Chef", "Puppet",
                            "Snowflake", "Excel", "Access", "Word", "Teams", "Sales", "Research", "Rust", "Dart",
                            "Julia", "Perl", "Scala", "Bash", "ML", "DL", "UI", "UX", "BI", "QA", "QC", "REST",
                            "SAP", "SAS", "Spark", "Node", "Outlook", "Looker", "Airflow", "Jenkins"}

# Short single-letter / two-letter language names that need list context.
LIST_CONTEXT_ONLY: set[str] = {"research", "sales", "go", "r", "c", "ml", "dl", "ui", "ux", "bi", "qa", "qc", "ts", "js", "rest", "sas"}

VENDOR_PREFIXES = ["Microsoft ", "Adobe ", "Apache ", "Autodesk ", "Intuit ", "IBM ", "Oracle ", "Google ", "Atlassian ",
                   "The MathWorks ", "Amazon Web Services AWS ", "Amazon ", "Apple ", "SAP ", "Red Hat ", "Salesforce.com ",
                   "Cisco ", "StataCorp ", "Esri ", "ESRI ", "Siemens ", "Dassault Systemes ", "Bentley ", "Hewlett-Packard HP "]

_ACRONYM_TAIL = re.compile(r"^(?P<long>[a-zA-Z][a-zA-Z ,\-/]+?)\s+(?P<acr>[A-Z][A-Z0-9/+#.]{1,9})$")


def derived_aliases(name: str) -> list[tuple[str, str]]:
    """Generate (alias, method) pairs for an O*NET technology name.

    * vendor prefix removal: 'Intuit QuickBooks' -> 'QuickBooks'
    * ' software' suffix removal for branded names: 'Workday software' -> 'Workday'
    * long-form/acronym split: 'Structured query language SQL' -> 'SQL', 'structured query language'
    """
    out: list[tuple[str, str]] = []
    for v in VENDOR_PREFIXES:
        if name.startswith(v) and len(name) > len(v) + 1:
            rest = name[len(v):]
            if rest[:1].isupper() or rest[:1].isdigit() or rest[:1] == ".":
                out.append((rest, "vendor_strip"))
    if name.endswith(" software"):
        rest = name[: -len(" software")]
        # only branded names ('Workday software', 'Microsoft Azure software'), not 'Web browser software'
        words = rest.split()
        brandish = (lambda w: any(c.isupper() for c in w[1:]) or any(c.isdigit() for c in w) or w.isupper())
        # only branded names ('Microsoft Azure software', 'FastEST software'), not 'Dictation software'
        if (len(words) == 1 and brandish(words[0])) or (len(words) > 1 and all(w[:1].isupper() or w[:1].isdigit() for w in words)):
            out.append((rest, "software_suffix_strip"))
            out.extend((a, f"software_suffix_strip+{m}") for a, m in derived_aliases(rest))
    m = _ACRONYM_TAIL.match(name)
    if m:
        long, acr = m.group("long").strip(), m.group("acr")
        initials = "".join(w[0] for w in re.split(r"[ \-/]", long) if w).upper()
        acr_letters = re.sub(r"[^A-Z]", "", acr)
        if len(acr_letters) >= 2 and (initials == acr_letters or initials.startswith(acr_letters) or acr_letters.startswith(initials) and len(initials) >= 2):
            out.append((acr, "acronym"))
            out.append((long, "long_form"))
    return out

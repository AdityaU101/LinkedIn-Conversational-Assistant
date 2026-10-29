// Every node also carries the shared :Node label so id lookups are always indexed.
CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;
// Uniqueness constraints (one per node label, keyed on the stable `id`).
CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (n:Candidate) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT job_id IF NOT EXISTS FOR (n:JobPosting) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT occupation_id IF NOT EXISTS FOR (n:Occupation) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (n:Skill) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (n:Requirement) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT experience_id IF NOT EXISTS FOR (n:Experience) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT credential_id IF NOT EXISTS FOR (n:Credential) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT project_id IF NOT EXISTS FOR (n:Project) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT education_id IF NOT EXISTS FOR (n:Education) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT location_id IF NOT EXISTS FOR (n:Location) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE n.id IS UNIQUE;
// Free-form SPO ablation graph (RQ3) lives under its own labels.
CREATE CONSTRAINT spo_entity_id IF NOT EXISTS FOR (n:SPOEntity) REQUIRE n.id IS UNIQUE;

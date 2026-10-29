# Ontology & knowledge graph

Definition: `ontology/ontology.json` (node types, relation domain/range, evidence requirements);
constraints: `ontology/constraints.cypher`.

## Nodes (stable IDs, `knowledge_graph/entity_linking.py`)

| Label | ID | Notes |
|---|---|---|
| Candidate | `cand:<candidate_id>` | |
| JobPosting | `job:<source>:<id>` | |
| Skill | `skill:<canonical_id>` | `onet-tech:<slug>`, `onet:<Element ID>`, `local:<slug>` |
| Occupation | `occ:<O*NET-SOC>` | |
| Company | `company:<slug>` | anonymised "Company Name" placeholders are not linked |
| Requirement, Experience, Education, Credential, Project | content-hash IDs | owned by their job / candidate |
| Location | `loc:<slug>` | |
| Evidence | `ev:<sha1(doc, start, end, text)>` | verbatim span with offsets, extractor, version, confidence |

Every node also carries `:Node` with a uniqueness constraint on `Node.id`, so every lookup is indexed.

## Relations

`HAS_SKILL, HELD, USED, AT, POSTED_BY, FOR_ROLE, HAS_REQUIREMENT, TARGETS, RELATED_TO, TYPICAL_FOR,
SUPPORTED_BY, HAS_CREDENTIAL, HAS_EDUCATION, WORKED_ON, LOCATED_IN`.

* Extracted relations carry `evidence_ids`; the subject has `SUPPORTED_BY` edges to those Evidence nodes.
* Skills a user adds in the profile editor are stored with `source="user_assertion"` and no evidence — they are
  never presented as resume evidence (explanations cite them as "added by the user").
* `RELATED_TO` between skills is **inferred** (`inferred=true`, `method`, `confidence`):
  `onet_occupation_cooccurrence_jaccard` (technologies listed by overlapping O*NET occupations, same
  technology category, Jaccard ≥ 0.30, top-8) and `embedding_cosine:all-MiniLM-L6-v2` (cosine ≥ 0.62, top-5,
  common skills only). Related ≠ identical: matching uses canonical IDs only.
* `RELATED_TO` between occupations and `TYPICAL_FOR` come from O*NET reference files.

## Writes

`GraphBatch.write` → batched `UNWIND … MERGE (n:Node {id}) SET n:<Label>` / `MERGE (a)-[:REL]->(b)`;
labels are whitelisted. Rebuilding is idempotent (tested). Candidate upserts delete the candidate's owned
facts first, then re-merge.

## Queries (`knowledge_graph/queries.py`)

candidate skills, job requirements, matched/missing, transferable bridges, related occupations,
explanation paths (Candidate→Skill←Requirement←Job, Skill-RELATED_TO bridges, shared Occupation),
visualisation subgraph, evidence lookup, and `consistency_report` (ontology conformance + evidence
completeness). Current development graph (4,296 jobs): 133,638 edges, 0 ontology violations, 100% of
evidence-required edges resolve to Evidence nodes — identical in Neo4j and in memory.

## Free-form SPO baseline (RQ3)

`knowledge_graph/spo.py` extracts subject–predicate–object triples from the same documents with spaCy
dependency parsing (and an LLM variant when a model is available). Experiment C compares the two graphs.

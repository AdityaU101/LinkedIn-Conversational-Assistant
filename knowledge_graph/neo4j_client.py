"""Graph stores: Neo4j (primary) and an in-memory fallback with the same API.

Only fixed, parameterised Cypher is executed. Labels and relationship types
are validated against ontology/ontology.json before being interpolated
(Cypher cannot parameterise labels); user input never reaches query text.
"""
from __future__ import annotations

import json
import logging
import pickle
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from common.config import PROJECT_ROOT, env, get_settings

log = logging.getLogger(__name__)
ONTOLOGY = json.loads((PROJECT_ROOT / "ontology" / "ontology.json").read_text())
LABELS = set(ONTOLOGY["nodes"]) | {"SPOEntity"}
REL_TYPES = set(ONTOLOGY["relations"]) | {"SPO"}


def _check_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Unknown node label {label!r}")
    return label


def _check_rel(rel: str) -> str:
    if rel not in REL_TYPES:
        raise ValueError(f"Unknown relationship type {rel!r}")
    return rel


class GraphStore:
    backend = "base"

    def merge_nodes(self, label: str, rows: list[dict]) -> None: ...
    def merge_edges(self, rel: str, rows: list[dict]) -> None: ...  # rows: {src, dst, props}
    def node(self, node_id: str) -> Optional[dict]: ...
    def edges(self, node_id: str, rel: Optional[str] = None, direction: str = "out") -> list[dict]: ...
    def delete_subject(self, node_id: str) -> int: ...
    def counts(self) -> dict: ...
    def all_edges(self, rel: Optional[str] = None) -> Iterable[dict]: ...
    def all_nodes(self, label: Optional[str] = None) -> Iterable[dict]: ...

    def neighbors(self, node_id: str, rel: Optional[str] = None, direction: str = "out") -> list[dict]:
        out = []
        for e in self.edges(node_id, rel, direction):
            other = e["dst"] if direction == "out" else e["src"]
            n = self.node(other)
            if n:
                out.append({"node": n, "edge": e})
        return out


class MemoryGraphStore(GraphStore):
    backend = "memory"

    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.nodes: dict[str, dict] = {}
        self.out: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
        self.inc: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
        self._lock = threading.RLock()
        if path and path.exists():
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            self.nodes, self.out, self.inc = d["nodes"], defaultdict(dict, d["out"]), defaultdict(dict, d["inc"])

    def save(self) -> None:
        if self.path:
            with self._lock, open(self.path, "wb") as fh:
                pickle.dump({"nodes": self.nodes, "out": dict(self.out), "inc": dict(self.inc)}, fh)

    def merge_nodes(self, label, rows):
        _check_label(label)
        with self._lock:
            for r in rows:
                cur = self.nodes.get(r["id"])
                if cur:
                    cur.update(r)
                else:
                    self.nodes[r["id"]] = {**r, "label": label}

    def merge_edges(self, rel, rows):
        _check_rel(rel)
        with self._lock:
            for r in rows:
                key = (rel, r["dst"])
                props = dict(r.get("props") or {})
                cur = self.out[r["src"]].get(key)
                if cur:
                    if "evidence_ids" in props:
                        props["evidence_ids"] = sorted(set(cur["props"].get("evidence_ids", [])) | set(props["evidence_ids"]))
                    cur["props"].update(props)
                else:
                    e = {"src": r["src"], "dst": r["dst"], "type": rel, "props": props}
                    self.out[r["src"]][key] = e
                    self.inc[r["dst"]][(rel, r["src"])] = e

    def node(self, node_id):
        return self.nodes.get(node_id)

    def edges(self, node_id, rel=None, direction="out"):
        table = self.out if direction == "out" else self.inc
        return [e for (t, _), e in table.get(node_id, {}).items() if rel is None or t == rel]

    def delete_subject(self, node_id):
        """Remove a node and everything only reachable through it as owned facts."""
        with self._lock:
            owned = {node_id}
            for e in self.edges(node_id):
                if e["type"] in {"HELD", "HAS_CREDENTIAL", "HAS_EDUCATION", "WORKED_ON", "SUPPORTED_BY", "HAS_REQUIREMENT"}:
                    owned.add(e["dst"])
                    owned.update(x["dst"] for x in self.edges(e["dst"], "SUPPORTED_BY"))
            for nid in owned:
                for e in list(self.out.get(nid, {}).values()):
                    self.inc[e["dst"]].pop((e["type"], nid), None)
                for e in list(self.inc.get(nid, {}).values()):
                    self.out[e["src"]].pop((e["type"], nid), None)
                self.out.pop(nid, None)
                self.inc.pop(nid, None)
                self.nodes.pop(nid, None)
            return len(owned)

    def counts(self):
        labels: dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            labels[n["label"]] += 1
        rels: dict[str, int] = defaultdict(int)
        for d in self.out.values():
            for (t, _) in d:
                rels[t] += 1
        return {"backend": self.backend, "nodes": dict(labels), "relationships": dict(rels)}

    def all_edges(self, rel=None):
        for d in list(self.out.values()):
            for (t, _), e in list(d.items()):
                if rel is None or t == rel:
                    yield e

    def all_nodes(self, label=None):
        for n in list(self.nodes.values()):
            if label is None or n["label"] == label:
                yield n


class Neo4jGraphStore(GraphStore):
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None, batch: int = 1000):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.database = database
        self.batch = batch
        self._ensure_constraints()

    def _run(self, q: str, **params):
        with self.driver.session(database=self.database) as s:
            return list(s.run(q, **params))

    def _ensure_constraints(self) -> None:
        text = (PROJECT_ROOT / "ontology" / "constraints.cypher").read_text()
        for stmt in [s.strip() for s in text.split(";")]:
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("//")]
            if "".join(lines).strip():
                self._run("\n".join(lines))

    def merge_nodes(self, label, rows):
        lab = _check_label(label)
        q = f"UNWIND $rows AS r MERGE (n:Node {{id: r.id}}) SET n:{lab} SET n += r"
        for i in range(0, len(rows), self.batch):
            self._run(q, rows=[{k: v for k, v in r.items() if v is not None} for r in rows[i:i + self.batch]])

    def merge_edges(self, rel, rows):
        t = _check_rel(rel)
        q = (f"UNWIND $rows AS r MATCH (a:Node {{id: r.src}}) MATCH (b:Node {{id: r.dst}}) "
             f"MERGE (a)-[e:{t}]->(b) SET e += r.props")
        for i in range(0, len(rows), self.batch):
            chunk = [{"src": r["src"], "dst": r["dst"],
                      "props": {k: v for k, v in (r.get("props") or {}).items() if v is not None}}
                     for r in rows[i:i + self.batch]]
            self._run(q, rows=chunk)

    def node(self, node_id):
        rec = self._run("MATCH (n:Node {id: $id}) RETURN n, [l IN labels(n) WHERE l <> 'Node'][0] AS label LIMIT 1",
                        id=node_id)
        return {**dict(rec[0]["n"]), "label": rec[0]["label"]} if rec else None

    def edges(self, node_id, rel=None, direction="out"):
        if rel:
            _check_rel(rel)
        pat = f"[e{':' + rel if rel else ''}]"
        q = (f"MATCH (a:Node {{id: $id}})-{pat}->(b) RETURN a.id AS src, b.id AS dst, type(e) AS type, properties(e) AS props"
             if direction == "out" else
             f"MATCH (b)-{pat}->(a:Node {{id: $id}}) RETURN b.id AS src, a.id AS dst, type(e) AS type, properties(e) AS props")
        return [dict(r) for r in self._run(q, id=node_id)]

    def delete_subject(self, node_id):
        q = ("MATCH (c:Node {id: $id}) OPTIONAL MATCH (c)-[:HELD|HAS_CREDENTIAL|HAS_EDUCATION|WORKED_ON|SUPPORTED_BY|HAS_REQUIREMENT]->(o) "
             "OPTIONAL MATCH (o)-[:SUPPORTED_BY]->(ev) WITH c, collect(DISTINCT o) + collect(DISTINCT ev) AS owned "
             "FOREACH (x IN owned | DETACH DELETE x) DETACH DELETE c RETURN size(owned) + 1 AS n")
        rec = self._run(q, id=node_id)
        return int(rec[0]["n"]) if rec else 0

    def counts(self):
        nodes = {r["l"]: r["c"] for r in self._run(
            "MATCH (n:Node) RETURN [l IN labels(n) WHERE l <> 'Node'][0] AS l, count(*) AS c")}
        rels = {r["t"]: r["c"] for r in self._run("MATCH ()-[e]->() RETURN type(e) AS t, count(*) AS c")}
        return {"backend": self.backend, "nodes": nodes, "relationships": rels}

    def all_edges(self, rel=None):
        if rel:
            _check_rel(rel)
        q = (f"MATCH (a)-[e{':' + rel if rel else ''}]->(b) "
             "RETURN a.id AS src, b.id AS dst, type(e) AS type, properties(e) AS props")
        return (dict(r) for r in self._run(q))

    def all_nodes(self, label=None):
        if label:
            _check_label(label)
        q = f"MATCH (n:{label or 'Node'}) RETURN n, [l IN labels(n) WHERE l <> 'Node'][0] AS label"
        return ({**dict(r["n"]), "label": r["label"]} for r in self._run(q))

    def save(self) -> None:  # Neo4j persists itself
        pass

    def close(self) -> None:
        self.driver.close()


def memory_path() -> Path:
    return get_settings().path("indexes") / "graph_memory.pkl"


@lru_cache(maxsize=1)
def get_graph_store() -> GraphStore:
    """Neo4j when configured and reachable (graph.backend auto|neo4j), else in-memory."""
    backend = get_settings().get("graph.backend", "auto")
    uri, user, pwd = env("NEO4J_URI"), env("NEO4J_USERNAME"), env("NEO4J_PASSWORD")
    if backend in ("auto", "neo4j") and uri and user and pwd:
        try:
            return Neo4jGraphStore(uri, user, pwd)
        except Exception as e:
            if backend == "neo4j":
                raise
            log.warning("Neo4j unavailable (%s); using in-memory graph store", type(e).__name__)
    return MemoryGraphStore(memory_path())

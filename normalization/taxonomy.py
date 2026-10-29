"""Build and load the skill/occupation taxonomy (O*NET primary + local supplement).

Build:  python scripts/build_taxonomy.py   (also run by scripts/build_indexes.py)
Outputs in Data/normalized/:
  skills.parquet             skill_id, name, skill_type, taxonomy, category_id, category_name, hot_technology, ...
  skill_aliases.parquet      alias, skill_id, method
  occupations.parquet        code, title, description, job_zone, major_group, major_group_title
  occupation_titles.parquet  code, title, title_source
  occupation_skills.parquet  code, skill_id, weight, relation_source
  related_occupations.parquet
  skill_relations.parquet    inferred RELATED_TO edges (inferred=true, method, confidence)
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd

from common.config import get_settings
from common.text import normalize_key, slug
from ingestion.onet_loader import ONetLoader
from normalization.aliases import ALIASES, LOCAL_SKILLS, derived_aliases

FILES = ["skills", "skill_aliases", "occupations", "occupation_titles", "occupation_skills",
         "related_occupations", "skill_relations"]


def out_dir() -> Path:
    return get_settings().path("normalized")


def local_skill_id(name: str) -> str:
    return "local:" + slug(name)


def build(onet: Optional[ONetLoader] = None, directory: Optional[Path] = None) -> dict:
    onet = onet or ONetLoader()
    directory = directory or out_dir()
    directory.mkdir(parents=True, exist_ok=True)
    s = get_settings()

    tech = onet.technology_skills()
    elems = onet.element_skills()
    local = pd.DataFrame({"skill_id": [local_skill_id(n) for n in LOCAL_SKILLS], "name": list(LOCAL_SKILLS),
                          "skill_type": "concept", "category_id": None, "category_name": list(LOCAL_SKILLS.values()),
                          "hot_technology": False, "in_demand": False, "taxonomy": "local"})
    skills = pd.concat([tech, elems, local], ignore_index=True)
    # The same element name can appear in two O*NET files (e.g. 'Mathematics' as skill and as
    # knowledge). Keep both IDs but resolve name lookups by the priority below.
    prio = {"technology": 0, "essential_skill": 1, "transferable_skill": 2, "concept": 3, "knowledge": 4}
    skills["name_priority"] = skills["skill_type"].map(prio)
    skills = skills.drop_duplicates(subset=["skill_id"])

    by_name: dict[str, str] = {}
    for r in skills.sort_values("name_priority").itertuples():
        by_name.setdefault(normalize_key(r.name), r.skill_id)

    alias_rows = []
    unresolved = []
    for alias, target in ALIASES.items():
        sid = by_name.get(normalize_key(target))
        if sid is None:
            unresolved.append((alias, target))
            continue
        alias_rows.append((alias, sid, "curated_alias"))
    for r in tech.itertuples():
        for alias, method in derived_aliases(r.name):
            if normalize_key(alias) not in by_name:  # never shadow an existing canonical name
                alias_rows.append((alias, r.skill_id, method))
    aliases = pd.DataFrame(alias_rows, columns=["alias", "skill_id", "method"])
    aliases["alias_key"] = aliases["alias"].map(normalize_key)
    # an alias that points to several different skills is ambiguous -> drop it
    # Curated aliases are authoritative. A *derived* alias pointing to several different
    # skills is ambiguous and dropped (unless a curated alias already decides it).
    curated = aliases[aliases["method"] == "curated_alias"]
    derived = aliases[(aliases["method"] != "curated_alias") & ~aliases["alias_key"].isin(set(curated["alias_key"]))]
    amb = derived.groupby("alias_key")["skill_id"].nunique()
    ambiguous = set(amb[amb > 1].index)
    derived = derived[~derived["alias_key"].isin(ambiguous)]
    aliases = pd.concat([curated, derived], ignore_index=True).drop_duplicates(subset=["alias_key"])

    occ = onet.occupations()
    titles = onet.occupation_titles()

    # Occupation -> skill weights: technologies (hot tech weighted higher) + element importance.
    ot = onet.occupation_technology()
    ot["weight"] = ot["hot_technology"].map({True: 1.0, False: 0.6})
    ot["relation_source"] = "onet_software_skills"
    imp = onet.occupation_element_importance()
    thr = s.get("graph.typical_skill_min_importance", 3.5)
    imp = imp[imp["importance"] >= thr].copy()
    imp["weight"] = (imp["importance"] / 5.0).round(3)
    imp["relation_source"] = "onet_" + imp["source_file"] + "_IM"
    occ_skills = pd.concat([ot[["code", "skill_id", "weight", "relation_source"]],
                            imp[["code", "skill_id", "weight", "relation_source"]]], ignore_index=True)
    occ_skills = occ_skills.sort_values("weight", ascending=False).drop_duplicates(subset=["code", "skill_id"])

    related = onet.related_occupations()
    skill_rel = infer_skill_relations(ot, tech, s.get("graph.related_skill_min_jaccard", 0.3),
                                      s.get("graph.related_skill_top_k", 8))
    emb_rel = infer_embedding_relations(skills)
    if len(emb_rel):
        skill_rel = pd.concat([skill_rel, emb_rel], ignore_index=True)

    skills.drop(columns=["name_priority"]).to_parquet(directory / "skills.parquet", index=False)
    aliases.to_parquet(directory / "skill_aliases.parquet", index=False)
    occ.to_parquet(directory / "occupations.parquet", index=False)
    titles.to_parquet(directory / "occupation_titles.parquet", index=False)
    occ_skills.to_parquet(directory / "occupation_skills.parquet", index=False)
    related.to_parquet(directory / "related_occupations.parquet", index=False)
    skill_rel.to_parquet(directory / "skill_relations.parquet", index=False)
    report = {"skills": int(len(skills)), "by_type": skills["skill_type"].value_counts().to_dict(),
              "aliases": int(len(aliases)), "alias_methods": aliases["method"].value_counts().to_dict(),
              "ambiguous_aliases_dropped": len(ambiguous), "unresolved_curated_aliases": unresolved,
              "occupations": int(len(occ)), "occupation_titles": int(len(titles)),
              "occupation_skill_links": int(len(occ_skills)), "related_occupation_links": int(len(related)),
              "inferred_skill_relations": int(len(skill_rel))}
    (directory / "taxonomy_report.json").write_text(json.dumps(report, indent=2))
    load_taxonomy.cache_clear()
    return report


def infer_skill_relations(occ_tech: pd.DataFrame, tech: pd.DataFrame, min_jaccard: float, top_k: int) -> pd.DataFrame:
    """RELATED_TO between technologies that co-occur across O*NET occupations.

    Jaccard over the sets of occupations that list each technology, restricted
    to the same O*NET technology category. Marked inferred=true; this is a
    relatedness signal, never an identity.
    """
    occs = occ_tech.groupby("skill_id")["code"].apply(set)
    occs = occs[occs.map(len) >= 3]
    cat = tech.set_index("skill_id")["category_id"]
    by_cat: dict[str, list[str]] = defaultdict(list)
    for sid in occs.index:
        by_cat[cat.get(sid)].append(sid)
    cand: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for members in by_cat.values():
        if len(members) > 600:  # avoid quadratic blow-up on giant generic categories
            members = sorted(members, key=lambda m: -len(occs[m]))[:600]
        for a, b in combinations(members, 2):
            sa, sb = occs[a], occs[b]
            inter = len(sa & sb)
            if not inter:
                continue
            j = inter / len(sa | sb)
            if j >= min_jaccard:
                cand[a].append((j, b))
                cand[b].append((j, a))
    rows = []
    for a, lst in cand.items():
        for j, b in sorted(lst, reverse=True)[:top_k]:
            if a < b or not any(x[1] == a for x in sorted(cand[b], reverse=True)[:top_k]):
                rows.append((a, b, round(j, 4)))
    df = pd.DataFrame(rows, columns=["skill_a", "skill_b", "confidence"]).drop_duplicates(subset=["skill_a", "skill_b"])
    df["inferred"] = True
    df["method"] = "onet_occupation_cooccurrence_jaccard"
    return df


def infer_embedding_relations(skills: pd.DataFrame, min_cos: float = 0.62, top_k: int = 5) -> pd.DataFrame:
    """RELATED_TO by sentence-embedding similarity between skill names.

    Restricted to commonly used skills (hot/in-demand technologies, local concepts and
    technologies used by >= 5 occupations). Skipped when no encoder is available.
    """
    from common.embeddings import get_encoder

    enc = get_encoder()
    cols = ["skill_id", "skill_a", "skill_b", "confidence", "inferred", "method"]
    if enc is None:
        return pd.DataFrame(columns=cols[1:])
    pool = skills[(skills["skill_type"] == "concept") | skills["hot_technology"].astype(bool)
                  | skills["in_demand"].astype(bool) | (skills.get("n_occupations", 0).fillna(0) >= 5)]
    pool = pool.drop_duplicates(subset=["skill_id"]).reset_index(drop=True)
    vecs = enc.encode_cached(pool["name"].tolist(), "skill_names_pool")
    sims = vecs @ vecs.T
    rows = []
    ids = pool["skill_id"].tolist()
    for i in range(len(ids)):
        sims[i, i] = -1
        for j in sims[i].argsort()[::-1][:top_k]:
            if sims[i, j] >= min_cos and ids[i] < ids[j]:
                rows.append((ids[i], ids[j], round(float(sims[i, j]), 4)))
    df = pd.DataFrame(rows, columns=["skill_a", "skill_b", "confidence"]).drop_duplicates()
    df["inferred"] = True
    df["method"] = "embedding_cosine:" + enc.model_name.split("/")[-1]
    return df


@dataclass
class Taxonomy:
    skills: pd.DataFrame
    aliases: pd.DataFrame
    occupations: pd.DataFrame
    occupation_titles: pd.DataFrame
    occupation_skills: pd.DataFrame
    related_occupations: pd.DataFrame
    skill_relations: pd.DataFrame

    def __post_init__(self):
        self.skill_by_id = self.skills.set_index("skill_id").to_dict("index")
        self.occ_by_code = self.occupations.set_index("code").to_dict("index")
        self.related_skills: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.relation_method: dict[tuple[str, str], str] = {}
        # a pair can be listed more than once (both directions, or by two methods): keep its strongest relation,
        # so a related skill is offered once, with the method that produced that confidence
        best: dict[tuple[str, str], tuple[float, str]] = {}
        for a, b, c, m in self.skill_relations[["skill_a", "skill_b", "confidence", "method"]].itertuples(index=False):
            key = (a, b) if a <= b else (b, a)
            if key not in best or float(c) > best[key][0]:
                best[key] = (float(c), m)
        for (a, b), (c, m) in best.items():
            self.related_skills[a].append((b, c))
            self.related_skills[b].append((a, c))
            self.relation_method[(a, b)] = self.relation_method[(b, a)] = m
        self.skills_by_occ: dict[str, dict[str, float]] = defaultdict(dict)
        for code, sid, w in self.occupation_skills[["code", "skill_id", "weight"]].itertuples(index=False):
            self.skills_by_occ[code][sid] = float(w)

    def skill_name(self, sid: str) -> str:
        return self.skill_by_id.get(sid, {}).get("name", sid)

    def occupation_title(self, code: str) -> str:
        return self.occ_by_code.get(code, {}).get("title", code)


@lru_cache(maxsize=1)
def load_taxonomy(directory: Optional[str] = None) -> Taxonomy:
    d = Path(directory) if directory else out_dir()
    if not (d / "skills.parquet").exists():
        build(directory=d)
    frames = {f: pd.read_parquet(d / f"{f}.parquet") for f in FILES}
    frames["aliases"] = frames.pop("skill_aliases")
    return Taxonomy(**frames)

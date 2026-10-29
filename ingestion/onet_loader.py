"""O*NET 31.0 loader — occupations, titles, skills, relations.

Identifiers are the real O*NET ones (O*NET-SOC codes, content-model Element
IDs). Individual technologies (``software_skills.Workplace Example``) have
no O*NET ID, so a stable derived ID ``onet-tech:<slug>`` is used and the
O*NET category Element ID is kept as ``category_id``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from common.config import get_settings
from common.text import slug


def onet_dir() -> Path:
    return get_settings().path("raw") / "onet"


def tech_skill_id(name: str) -> str:
    return "onet-tech:" + slug(name, 120)


class ONetLoader:
    def __init__(self, base: Optional[Path] = None):
        self.base = base or onet_dir()

    def _read(self, name: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self.base / f"{name}.csv", low_memory=False, **kw)

    def occupations(self) -> pd.DataFrame:
        occ = self._read("occupation_data").rename(
            columns={"O*NET-SOC Code": "code", "Title": "title", "Description": "description"})
        jz = self._read("job_zones").rename(columns={"O*NET-SOC Code": "code", "Job Zone": "job_zone"})
        occ = occ.merge(jz[["code", "job_zone"]], on="code", how="left")
        occ["major_group"] = occ["code"].str[:2]
        majors = self.soc_major_groups()
        occ["major_group_title"] = occ["major_group"].map(majors)
        return occ

    def soc_major_groups(self) -> dict[str, str]:
        """SOC major group names from OSMT's BLS-Import.csv (falls back to codes)."""
        bls = get_settings().path("raw") / "osmt" / "import" / "BLS-Import.csv"
        if not bls.exists():
            return {}
        df = pd.read_csv(bls)
        df = df[df["SOC Group"] == "Major"]
        return {c[:2]: t for c, t in zip(df["SOC Code"], df["SOC Title"])}

    def occupation_titles(self) -> pd.DataFrame:
        """All known titles per occupation: canonical, alternate (job_titles) and reported titles."""
        occ = self._read("occupation_data")[["O*NET-SOC Code", "Title"]]
        frames = [pd.DataFrame({"code": occ["O*NET-SOC Code"], "title": occ["Title"], "title_source": "occupation"})]
        jt = self._read("job_titles")
        frames.append(pd.DataFrame({"code": jt["O*NET-SOC Code"], "title": jt["Job Title"], "title_source": "alternate"}))
        short = jt.dropna(subset=["Short Title"])
        frames.append(pd.DataFrame({"code": short["O*NET-SOC Code"], "title": short["Short Title"], "title_source": "short"}))
        rt = self._read("sample_of_reported_titles")
        frames.append(pd.DataFrame({"code": rt["O*NET-SOC Code"], "title": rt["Reported Job Title"],
                                    "title_source": "reported"}))
        out = pd.concat(frames, ignore_index=True).dropna(subset=["title"])
        return out.drop_duplicates(subset=["code", "title"])

    def technology_skills(self) -> pd.DataFrame:
        sw = self._read("software_skills")
        g = sw.groupby("Workplace Example").agg(
            category_id=("Element ID", "first"), category_name=("Element Name", "first"),
            hot_technology=("Hot Technology", lambda s: bool((s == "Y").any())),
            in_demand=("In Demand", lambda s: bool((s == "Y").any())),
            n_occupations=("O*NET-SOC Code", "nunique"),
        ).reset_index().rename(columns={"Workplace Example": "name"})
        g["skill_id"] = g["name"].map(tech_skill_id)
        g["skill_type"] = "technology"
        g["taxonomy"] = "onet"
        return g

    def element_skills(self) -> pd.DataFrame:
        """Essential skills, transferable skills and knowledge areas (content-model elements)."""
        frames = []
        for name, stype in [("essential_skills", "essential_skill"), ("transferable_skills", "transferable_skill"),
                            ("knowledge", "knowledge")]:
            df = self._read(name, usecols=["Element ID", "Element Name"]).drop_duplicates()
            frames.append(pd.DataFrame({"skill_id": "onet:" + df["Element ID"], "name": df["Element Name"],
                                        "skill_type": stype, "category_id": df["Element ID"].str.rsplit(".", n=1).str[0],
                                        "category_name": stype.replace("_", " "), "hot_technology": False,
                                        "in_demand": False, "taxonomy": "onet"}))
        ref = self._read("content_model_reference").rename(columns={"Element ID": "eid", "Description": "description"})
        out = pd.concat(frames, ignore_index=True)
        out["description"] = out["skill_id"].str[5:].map(ref.set_index("eid")["description"])
        return out

    def occupation_technology(self) -> pd.DataFrame:
        sw = self._read("software_skills")
        return pd.DataFrame({"code": sw["O*NET-SOC Code"], "skill_id": sw["Workplace Example"].map(tech_skill_id),
                             "hot_technology": sw["Hot Technology"] == "Y", "in_demand": sw["In Demand"] == "Y"})

    def occupation_element_importance(self) -> pd.DataFrame:
        """Importance (scale IM, 1–5) of each skill/knowledge element per occupation."""
        frames = []
        for name in ("essential_skills", "transferable_skills", "knowledge"):
            df = self._read(name, usecols=["O*NET-SOC Code", "Element ID", "Scale ID", "Data Value", "Recommend Suppress"])
            df = df[(df["Scale ID"] == "IM") & (df["Recommend Suppress"] != "Y")]
            frames.append(pd.DataFrame({"code": df["O*NET-SOC Code"], "skill_id": "onet:" + df["Element ID"],
                                        "importance": df["Data Value"].astype(float), "source_file": name}))
        return pd.concat(frames, ignore_index=True)

    def related_occupations(self) -> pd.DataFrame:
        df = self._read("related_occupations")
        return pd.DataFrame({"code": df["O*NET-SOC Code"], "related_code": df["Related O*NET-SOC Code"],
                             "tier": df["Relatedness Tier"], "rank": df["Index"]})

    def task_text(self) -> pd.DataFrame:
        ts = self._read("task_statements", usecols=["O*NET-SOC Code", "Task"])
        return ts.groupby("O*NET-SOC Code")["Task"].apply(lambda s: " ".join(s)).reset_index().rename(
            columns={"O*NET-SOC Code": "code", "Task": "tasks"})

    def education_levels(self) -> pd.DataFrame:
        """Most common required education category per occupation (RL scale)."""
        ed = self._read("education", usecols=["O*NET-SOC Code", "Scale ID", "Category", "Data Value"])
        ed = ed[ed["Scale ID"] == "RL"]
        idx = ed.groupby("O*NET-SOC Code")["Data Value"].idxmax()
        return ed.loc[idx, ["O*NET-SOC Code", "Category"]].rename(
            columns={"O*NET-SOC Code": "code", "Category": "education_category"})


@lru_cache(maxsize=1)
def get_onet() -> ONetLoader:
    return ONetLoader()

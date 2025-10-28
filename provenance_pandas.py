from __future__ import annotations
import hashlib
import json
import uuid
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, PROV, XSD, DCTERMS
    _HAVE_RDFLIB = True
except Exception:
    _HAVE_RDFLIB = False


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hashMaker_df(df: pd.DataFrame, primary_key: Optional[List[str]] = None, sample_rows: Optional[int] = None) -> str:
    """
    Compute a stable-ish fingerprint of a DataFrame content, using hash_pandas_object.
    If a primary_key is provided and present, sort by it for stability.
    """
    work = df.copy()

    # Ensure column order
    work = work.reindex(sorted(work.columns), axis=1)

    if primary_key and all(k in work.columns for k in primary_key):
        work = work.sort_values(by=primary_key)
    else:
        # stable sort by index for consistency
        work = work.sort_index()

    if sample_rows is not None and len(work) > sample_rows:
        work = work.iloc[:sample_rows]

    # Normalize missing markers for hashing
    work = work.replace({" ?": np.nan, "?": np.nan, "NA": np.nan, "N/A": np.nan, "nan": np.nan, "NULL": np.nan})

    # Convert to bytes via pandas hashing utilities
    h = pd.util.hash_pandas_object(work, index=True).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


def missing(df: pd.DataFrame) -> Dict[str, Any]:
    nulls = df.isna().sum().to_dict()
    # treat '?' variants as missing-like signals
    qmask = df.isin(["?", " ?","NA","N/A","nan","NULL"]).sum().to_dict()
    return {"na_by_col": nulls, "question_mark_like_by_col": qmask}


def compare(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # align indexes and columns for fair comparison
    common_cols = sorted(set(a.columns).intersection(set(b.columns)))
    a2 = a.reindex(columns=common_cols).sort_index()
    b2 = b.reindex(columns=common_cols).sort_index()
    common_idx = a2.index.union(b2.index)
    return a2.reindex(common_idx), b2.reindex(common_idx)


@dataclass
class OpRecord:
    op_id: str
    op_name: str
    timestamp: str
    parameters: Dict[str, Any]
    pre_shape: Tuple[int, int]
    post_shape: Tuple[int, int]
    pre_columns: List[str]
    post_columns: List[str]
    pre_hash: str
    post_hash: str
    deltas: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    code_context: Dict[str, Any] = field(default_factory=dict)


class ProvenanceTracker:
    def __init__(self, dataset_name: str, primary_key: Optional[List[str]] = None, agent: Optional[str] = None):
        self.dataset_name = dataset_name
        self.primary_key = primary_key
        self.agent = agent or "unknown_user"
        self.records: List[OpRecord] = []
        self.run_id = str(uuid.uuid4())
        self.created_at = now_iso()

    def log_operation(self, before: pd.DataFrame, after: pd.DataFrame, op_name: str, parameters: Dict[str, Any], code_context: Optional[Dict[str, Any]] = None):
        pre_hash = hashMaker_df(before, self.primary_key)
        post_hash = hashMaker_df(after, self.primary_key)

        pre_shape = before.shape
        post_shape = after.shape
        pre_columns = list(before.columns)
        post_columns = list(after.columns)

        # Diff summaries
        deltas: Dict[str, Any] = {}
        deltas["row_count_change"] = post_shape[0] - pre_shape[0]
        deltas["column_added"] = sorted(list(set(post_columns) - set(pre_columns)))
        deltas["column_removed"] = sorted(list(set(pre_columns) - set(post_columns)))

        a_aligned, b_aligned = compare(before, after)
        try:
            changed_cells = (a_aligned != b_aligned).sum().sum()
        except Exception:
            changed_cells = None
        deltas["changed_cells_on_common"] = int(changed_cells) if changed_cells is not None else None

        # Get back data quality metrics
        data_quality = {
            "pre_missing": missing(before),
            "post_missing": missing(after),
        }

        rec = OpRecord(
            op_id=str(uuid.uuid4()),
            op_name=op_name,
            timestamp=now_iso(),
            parameters=parameters,
            pre_shape=pre_shape,
            post_shape=post_shape,
            pre_columns=pre_columns,
            post_columns=post_columns,
            pre_hash=pre_hash,
            post_hash=post_hash,
            deltas=deltas,
            data_quality=data_quality,
            code_context=code_context or {},
        )
        self.records.append(rec)

    ### EXPORT METHODS ###
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "primary_key": self.primary_key,
            "agent": self.agent,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "records": [asdict(r) for r in self.records],
        }

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_graph(self, path: str):
        """
        A simple DOT graph representing the sequence of operations.
        Entities/values for graph -- hashes and operations.
        """
        lines = ["dgraph provenance {", '  rankdir=LR;', '  node [shape=box];']
        prev_hash = None
        step = 0
        for r in self.records:
            inp = r.pre_hash[:8]
            out = r.post_hash[:8]
            # entities
            lines.append(f'  "E_{inp}" [label="Entity {inp}\\n{r.pre_shape[0]}x{r.pre_shape[1]}"];')
            lines.append(f'  "E_{out}" [label="Entity {out}\\n{r.post_shape[0]}x{r.post_shape[1]}"];')
            # activity
            lines.append('  node [shape=ellipse];')
            act_id = f"A_{step}"
            params_preview = ", ".join(f"{k}={str(v)[:12]}" for k,v in r.parameters.items())
            lines.append(f'  "{act_id}" [label="{r.op_name}\\n{params_preview}"];')
            # edges
            lines.append('  node [shape=box];')
            lines.append(f'  "E_{inp}" -> "{act_id}" [label="used"];')
            lines.append(f'  "{act_id}" -> "E_{out}" [label="wasGeneratedBy"];')
            prev_hash = r.post_hash
            step += 1
        lines.append("}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def save_rdf(self, path: str):
        if not _HAVE_RDFLIB:
            raise ImportError("rdflib not installed")
        g = Graph()
        PROJ = Namespace("http://example.org/prov/")
        g.bind("prov", PROV)
        g.bind("dct", DCTERMS)
        g.bind("proj", PROJ)

        run_uri = URIRef(f"{PROJ}run/{self.run_id}")
        g.add((run_uri, RDF.type, PROV.Activity))
        g.add((run_uri, DCTERMS.creator, Literal(self.agent)))
        g.add((run_uri, DCTERMS.created, Literal(self.created_at, datatype=XSD.dateTime)))

        for idx, r in enumerate(self.records):
            act = URIRef(f"{PROJ}activity/{r.op_id}")
            ent_in = URIRef(f"{PROJ}entity/{r.pre_hash}")
            ent_out = URIRef(f"{PROJ}entity/{r.post_hash}")

            g.add((act, RDF.type, PROV.Activity))
            g.add((ent_in, RDF.type, PROV.Entity))
            g.add((ent_out, RDF.type, PROV.Entity))

            g.add((act, DCTERMS.title, Literal(r.op_name)))
            g.add((act, DCTERMS.created, Literal(r.timestamp, datatype=XSD.dateTime)))

            g.add((act, PROV.used, ent_in))
            g.add((ent_out, PROV.wasGeneratedBy, act))
            g.add((act, PROV.wasInformedBy, run_uri))

        g.serialize(destination=path, format="turtle")


class ProvenanceDataFrame:
    """
    Simple wrapper to log prov for common cleaning operaration (predefined).
    """
    def __init__(self, df: pd.DataFrame, tracker: ProvenanceTracker, code_context: Optional[Dict[str, Any]] = None):
        self.df = df
        self.tracker = tracker
        self.code_context = code_context or {}

    def _wrap(self, op_name: str, parameters: Dict[str, Any], func: Callable[[pd.DataFrame], pd.DataFrame]) -> "ProvenanceDataFrame":
        before = self.df
        after = func(before.copy())
        self.tracker.log_operation(before, after, op_name=op_name, parameters=parameters, code_context=self.code_context)
        return ProvenanceDataFrame(after, self.tracker, self.code_context)

############COMMON OPERATIONS############
    def drop_columns(self, columns: List[str]) -> "ProvenanceDataFrame":
        return self._wrap("drop_columns", {"columns": columns}, lambda d: d.drop(columns=columns))
    #### FROM STATIC POV these have type errors but are valid ####
    def dropna(self, *args, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("dropna", {"args": list(args), "kwargs": kwargs}, lambda d: d.dropna(*args, **kwargs))

    def fillna(self, *args, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("fillna", {"args": list(args), "kwargs": kwargs}, lambda d: d.fillna(*args, **kwargs))

    def rename(self, *args, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("rename", {"args": list(args), "kwargs": kwargs}, lambda d: d.rename(*args, **kwargs))

    def merge(self, right: "ProvenanceDataFrame|pd.DataFrame", *args, **kwargs) -> "ProvenanceDataFrame":
        if isinstance(right, ProvenanceDataFrame):
            right_df = right.df
        else:
            right_df = right
        return self._wrap("merge", {"args": list(args), "kwargs": kwargs}, lambda d: d.merge(right_df, *args, **kwargs))

    def append(self, right: "ProvenanceDataFrame|pd.DataFrame", **kwargs) -> "ProvenanceDataFrame":
        # Modern append via concat for row addition
        if isinstance(right, ProvenanceDataFrame):
            right_df = right.df
        else:
            right_df = right
        return self._wrap("append", {"kwargs": kwargs}, lambda d: pd.concat([d, right_df], **kwargs))
    

    ###SEPATRATE METHOD FOR APPLY FUNCTION####
    def apply(self, func: Callable[[pd.DataFrame], pd.DataFrame], name: str, params: Optional[Dict[str, Any]] = None) -> "ProvenanceDataFrame":
        params = params or {}
        return self._wrap(name, params, func)

    ### EXPORT TO PANDAS ###
    def to_pandas(self) -> pd.DataFrame:
        return self.df.copy()
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
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, PROV, XSD, DCTERMS
HAVE_RDFLIB = True
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
    lineage: Optional[List[str]] = None
    samples: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ProvenanceTracker:
    def __init__(self, dataset_name: str, primary_key: Optional[List[str]] = None, agent: Optional[str] = None):
        self.dataset_name = dataset_name
        self.primary_key = primary_key
        self.agent = agent or "unknown_user"
        self.records: List[OpRecord] = []
        self.run_id = str(uuid.uuid4())
        self.created_at = now_iso()
        self.entity_map: Dict[str, Any] = {}

    def grab_samples(self, df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
        if len(df) <= n:
            return df.to_dict(orient='records')
        
        h = df.head(n // 2 + 1)
        t = df.tail(n // 2)
        samp = pd.concat([h, t])
        samp = samp[~samp.index.duplicated()]
        
        return json.loads(samp.to_json(orient='records', date_format='iso'))

    def find_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        out = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
                cnt = ((df[col] < low) | (df[col] > high)).sum()
                if cnt > 0:
                    out[col] = int(cnt)
        return out

    def check_types(self, df: pd.DataFrame, after: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        types = {}
        for col in df.columns:
            dt = str(df[col].dtype)
            if after is not None and col in after.columns:
                prev = dt
                new = str(after[col].dtype)
                types[col] = (prev, new) if prev != new else None
            else:
                 types[col] = dt
        return types

    def log_operation(self, before: pd.DataFrame, after: pd.DataFrame, op_name: str, parameters: Dict[str, Any], code_context: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
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
            total_cells = a_aligned.size
            delta_percent = (changed_cells / total_cells * 100) if total_cells > 0 else 0
        except Exception:
            changed_cells = None
            delta_percent = 0
        deltas["changed_cells_on_common"] = int(changed_cells) if changed_cells is not None else None
        deltas["delta_percentage"] = delta_percent

        # Get back data quality metrics
        data_quality = {
            "pre_missing": missing(before),
            "post_missing": missing(after),
            "data_type_changes": self.check_types(before, after),
            "outliers_before": self.find_outliers(before),
            "outliers_after": self.find_outliers(after)
        }
        
        lineage = [r.post_hash for r in self.records if r.post_hash != pre_hash]
        
        samples = {
            "before": self.grab_samples(before),
            "after": self.grab_samples(after)
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
            lineage=lineage,
            samples=samples,
            error=error
        )
        self.records.append(rec)
        self.entity_map[post_hash] = rec

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
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_graph(self, path: str):
        """
        A simple DOT graph representing the sequence of operations.
        Entities/values for graph -- hashes and operations.
        """
        try:
            lines = ["digraph provenance {", '  rankdir=LR;', '  node [shape=box];']
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
                step += 1
            lines.append("}")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            print(f"couldn't write graph: {e}")

    def save_rdf(self, path: str):
        if not HAVE_RDFLIB:
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
    
    def replace(self, *args, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("replace", {"args": list(args), "kwargs": kwargs}, lambda d: d.replace(*args, **kwargs))
        
    def astype(self, *args, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("astype", {"args": list(args), "kwargs": kwargs}, lambda d: d.astype(*args, **kwargs))

    def apply(self, func: Callable[[pd.DataFrame], pd.DataFrame], name: str, params: Optional[Dict[str, Any]] = None) -> "ProvenanceDataFrame":
        params = params or {}
        return self._wrap(name, params, func)

    def drop_duplicates(self, subset: Optional[List[str]] = None, keep: str = "first", **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("drop_duplicates", {"subset": subset, "keep": keep, "kwargs": kwargs}, 
                        lambda d: d.drop_duplicates(subset=subset, keep=keep, **kwargs))

    def sort_values(self, by, asc: bool = True, *args, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("sort_values", {"by": by, "ascending": asc, "args": list(args), "kwargs": kwargs}, 
                        lambda d: d.sort_values(by=by, ascending=asc, *args, **kwargs))

    def sample_rows(self, n: Optional[int] = None, frac: Optional[float] = None, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("sample_rows", {"n": n, "frac": frac, "kwargs": kwargs}, 
                        lambda d: d.sample(n=n, frac=frac, **kwargs))

    def select_columns(self, cols: List[str]) -> "ProvenanceDataFrame":
        return self._wrap("select_columns", {"columns": cols}, lambda d: d[cols])

    def assign(self, **kwargs) -> "ProvenanceDataFrame":
        p = {"columns": {k: getattr(v, "__name__", str(v)[:50]) if callable(v) else v for k, v in kwargs.items()}}
        return self._wrap("assign", p, lambda d: d.assign(**kwargs))

    def groupby_agg(self, by, agg: Dict[str, Any], as_index: bool = False) -> "ProvenanceDataFrame":
        p = {"by": by, "agg": agg, "as_index": as_index}
        return self._wrap("groupby_agg", p, lambda d: d.groupby(by=by, as_index=as_index).agg(agg))

    def head(self, n: int = 5) -> "ProvenanceDataFrame":
        return self._wrap("head", {"n": n}, lambda d: d.head(n))

    def tail(self, n: int = 5) -> "ProvenanceDataFrame":
        return self._wrap("tail", {"n": n}, lambda d: d.tail(n))

    def reset_index(self, drop: bool = True, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("reset_index", {"drop": drop, "kwargs": kwargs}, 
                        lambda d: d.reset_index(drop=drop, **kwargs))

    def set_index(self, keys, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("set_index", {"keys": keys, "kwargs": kwargs}, 
                        lambda d: d.set_index(keys=keys, **kwargs))

    def melt(self, id_vars: Optional[List[str]] = None, value_vars: Optional[List[str]] = None, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("melt", {"id_vars": id_vars, "value_vars": value_vars, "kwargs": kwargs}, 
                        lambda d: d.melt(id_vars=id_vars, value_vars=value_vars, **kwargs))

    def pivot(self, index, columns, values: Optional[str] = None, **kwargs) -> "ProvenanceDataFrame":
        return self._wrap("pivot", {"index": index, "columns": columns, "values": values, "kwargs": kwargs}, 
                        lambda d: d.pivot(index=index, columns=columns, values=values, **kwargs))

    def str_replace(self, col: str, pat: str, repl: str, regex: bool = False, **kwargs) -> "ProvenanceDataFrame":
        p = {"column": col, "pat": pat, "repl": repl, "regex": regex, "kwargs": kwargs}
        
        def do_replace(d: pd.DataFrame) -> pd.DataFrame:
            w = d.copy()
            w[col] = w[col].str.replace(pat, repl, regex=regex, **kwargs)
            return w
            
        return self._wrap("str_replace", p, do_replace)

    def str_strip(self, col: str, side: str = "both", **kwargs) -> "ProvenanceDataFrame":
        p = {"column": col, "side": side, "kwargs": kwargs}
        
        def do_strip(d: pd.DataFrame) -> pd.DataFrame:
            w = d.copy()
            w[col] = w[col].str.strip(**kwargs)
            return w
            
        return self._wrap("str_strip", p, do_strip)

    def lower(self, cols: Optional[List[str]] = None) -> "ProvenanceDataFrame":
        p = {"columns": cols}
        
        def do_lower(d: pd.DataFrame) -> pd.DataFrame:
            w = d.copy()
            c = cols or [col for col in w.columns if pd.api.types.is_object_dtype(w[col])]
            for col in c:
                if col in w.columns:
                    w[col] = w[col].str.lower()
            return w
            
        return self._wrap("lower", p, do_lower)

    def upper(self, cols: Optional[List[str]] = None) -> "ProvenanceDataFrame":
        p = {"columns": cols}
        
        def do_upper(d: pd.DataFrame) -> pd.DataFrame:
            w = d.copy()
            c = cols or [col for col in w.columns if pd.api.types.is_object_dtype(w[col])]
            for col in c:
                if col in w.columns:
                    w[col] = w[col].str.upper()
            return w
            
        return self._wrap("upper", p, do_upper)

    def get_duplicates(self, subset: Optional[List[str]] = None, keep = False) -> pd.DataFrame:
        return self.df.duplicated(subset=subset, keep=keep)

    def describe_summary(self) -> Dict[str, Any]:
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing": self.df.isna().sum().to_dict(),
            "memory_usage": self.df.memory_usage(deep=True).sum(),
            "numeric_stats": self.df.describe().to_dict(),
        }

    ### EXPORT TO PANDAS ###
    def to_pandas(self) -> pd.DataFrame:
        return self.df.copy()
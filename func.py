########## WORKING EXAMPLE -- use CALL to repurposedrugs
from __future__ import annotations

import re
from typing import Iterable, Union, Optional, List, Dict, Any
import requests
import pandas as pd

BASE_URL = "https://repurposedrugs.aittokallio.group"
NCT_RE = re.compile(r"NCT\d{8}", re.IGNORECASE)

def _extract_nct_ids(ref_url: Optional[str]) -> List[str]:
    if not ref_url:
        return []
    ids = [x.upper() for x in NCT_RE.findall(ref_url)]
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _as_list(x: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return list(x)

def _is_whole_number(x: float, tol: float = 1e-9) -> bool:
    return x is not None and abs(x - round(x)) < tol

def query_pair_raw(drug: str, disease: str, base_url: str = BASE_URL, timeout_s: int = 60) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/readjson_new.php"
    r = requests.get(url, params={"drugs": drug, "diseases": disease}, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def repurposedrugs_table(
    drugs: Union[str, Iterable[str]],
    diseases: Union[str, Iterable[str]],
    base_url: str = BASE_URL,
    timeout_s: int = 60,
) -> pd.DataFrame:
    drugs_l = _as_list(drugs)
    diseases_l = _as_list(diseases)
    rows = []
    for drug in drugs_l:
        for disease in diseases_l:
            obj = query_pair_raw(drug, disease, base_url=base_url, timeout_s=timeout_s)
            val: Optional[float]
            try:
                val = float(obj["data"][0])
            except Exception:
                val = None
            info = (obj.get("druginfo") or [])
            nct_ids: List[str] = []
            phase_from_info: Optional[int] = None
            if info:
                try:
                    phase_from_info = int(info[0].get("Phase"))
                except Exception:
                    phase_from_info = None
                nct_ids = _extract_nct_ids(info[0].get("Merged_RefNew"))
            has_nct = len(nct_ids) > 0
            phase: Optional[int] = None
            pred_score: Optional[float] = None
            if val is not None and _is_whole_number(val) and has_nct:
                phase = int(round(val)) if phase_from_info is None else phase_from_info # phase
                pred_score = None
            else:
                pred_score = val #pred score
                phase = None
            rows.append(
                {
                    "drug": drug,
                    "disease": disease,
                    "prediction_score": pred_score,
                    "phase": phase,
                    "nct_id": ",".join(nct_ids) if nct_ids else None
                }
            )
    return pd.DataFrame(rows, columns=["drug", "disease", "prediction_score", "phase", "nct_id"])


df = repurposedrugs_table(
    drugs=["Methotrexate","Ibrutinib","Aspirin"],
    diseases=["Chronic Lymphocytic Leukemia", "Bipolar Disorder", "Hepatitis c"],
)
print(df)


### ML SMILES 
import json
import re
import requests
import pandas as pd
from urllib.parse import quote

BASE_URL = "https://repurposedrugs.aittokallio.group"
_JSON_RE = re.compile(r"\{.*\}\s*$", re.DOTALL)

def _looks_like_smiles(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    tokens = ["=", "#", "(", ")", "[", "]", "@", "\\", "/", "+", "-", "%"]
    if any(t in s for t in tokens):
        return True
    if any(ch.isdigit() for ch in s):  # ring closures
        return True
    return False

def _pubchem_name_to_smiles(name: str, timeout_s: int = 30) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Empty name cannot be resolved via PubChem.")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name)}/property/IsomericSMILES/JSON"
    )
    r = requests.get(url, timeout=timeout_s)
    if r.status_code == 404:
        raise ValueError(f"PubChem could not find a compound for name: {name!r}")
    r.raise_for_status()
    data = r.json()
    try:
        return data["PropertyTable"]["Properties"][0]["SMILES"]
    except Exception as e:
        raise ValueError(f"Unexpected PubChem response for {name!r}: {e}")

def _call_runqc(name: str, smiles: str, timeout_s: int = 120) -> dict:
    url = f"{BASE_URL.rstrip('/')}/runQC.php"
    r = requests.get(url, params={"name": name, "smiles": smiles}, timeout=timeout_s)
    r.raise_for_status()
    text = r.text.strip()
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("Could not find JSON in response. First 1200 chars:\n" + text[:1200])
    return json.loads(m.group(0))

def predict_repurposedrugs(
    query: str,
    *,
    name: str | None = None,
    base_url: str = BASE_URL,
    timeout_pubchem_s: int = 30,
    timeout_runqc_s: int = 120,
) -> pd.DataFrame:
    q = query.strip()
    if not q:
        raise ValueError("query is empty")
    if _looks_like_smiles(q):
        smiles = q
        drug_label = name if name else "custom"
    else:
        drug_label = name if name else q
        smiles = _pubchem_name_to_smiles(q, timeout_s=timeout_pubchem_s)
    payload = _call_runqc(drug_label, smiles, timeout_s=timeout_runqc_s)
    diseases = payload.get("diseases", [])
    values = payload.get("values", [])
    if len(diseases) != len(values):
        raise ValueError(f"Length mismatch: diseases={len(diseases)} values={len(values)}")
    df = pd.DataFrame(
        {
            "drug": [drug_label] * len(diseases),
            "smiles": [smiles] * len(diseases),
            "disease": diseases,
            "prediction_score": values
        }
    ).sort_values("prediction_score", ascending=False, ignore_index=True)
    return df

df = predict_repurposedrugs("Metformin")
print(df)

df = predict_repurposedrugs("CN(C)C(=N)N=C(N)N")
print(df)

######## get the ncids and the phase from xlsx file
from __future__ import annotations
import re
from typing import Iterable, Union, Optional, List, Dict, Any
import pandas as pd


_PHASE_RE = re.compile(r"^\s*Phase\s*(\d+)\s*$", re.IGNORECASE)
NCT_RE = re.compile(r"NCT\d{8}", re.IGNORECASE)

def _parse_cell_to_phase_or_score(x) -> tuple[Optional[float], Optional[int]]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return (None, None)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return (None, None)
        m = _PHASE_RE.match(s)
        if m:
            return (None, int(m.group(1)))
        try:
            return (float(s), None)
        except Exception:
            return (None, None)
    if isinstance(x, (int, float)):
        return (float(x), None)
    return (None, None)


def sparse_xlsx_to_table(
    xlsx_path: str,
    sheet_name=0,
    *,
    header_row: int = 0,        # row with disease names
    group_row: int = 1,         # row with disease groups
    first_data_row: int = 2,    # first row containing drugs
    drug_col: int = 0,          # column with drug names
    keep_disease_group: bool = True,
) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    diseases = raw.iloc[header_row, drug_col + 1 :].tolist()
    disease_groups = raw.iloc[group_row, drug_col + 1 :].tolist()
    drugs = raw.iloc[first_data_row:, drug_col].astype(str).tolist()
    mat = raw.iloc[first_data_row:, drug_col + 1 : drug_col + 1 + len(diseases)].copy()
    out_rows = []
    for i, drug in enumerate(drugs):
        drug_s = str(drug).strip()
        if drug_s == "" or drug_s.lower() == "nan":
            continue
        for j, disease in enumerate(diseases):
            disease_s = str(disease).strip() if pd.notna(disease) else ""
            if disease_s == "":
                continue
            cell = mat.iat[i, j]
            pred, phase = _parse_cell_to_phase_or_score(cell)
            if pred is None and phase is None:
                continue  # keep sparse
            row = {
                "drug": drug_s,
                "disease": disease_s,
                "prediction_score": pred,
                "phase": phase,
                "nct_id": None,
            }
            if keep_disease_group:
                dg = disease_groups[j]
                row["disease_group"] = None if pd.isna(dg) else str(dg).strip()
            out_rows.append(row)
    cols = ["drug", "disease", "prediction_score", "phase", "nct_id"]
    if keep_disease_group:
        cols.append("disease_group")
    return pd.DataFrame(out_rows, columns=cols)


def extract_nct_ids(x) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    s = str(x)
    ids = [m.upper() for m in NCT_RE.findall(s)]
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def load_nct_map_from_sheet(
    xlsx_path: str,
    sheet_name=0,
    drug_col: Optional[str] = None,
    disease_col: Optional[str] = None,
    ref_col: Optional[str] = None,
) -> pd.DataFrame:
    nct_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    if drug_col is None or disease_col is None or ref_col is None:
        cols = {c.lower(): c for c in nct_raw.columns}
        drug_col = drug_col or next((cols[k] for k in ["drug_name", "drug", "compound", "drugname"] if k in cols), None)
        disease_col = disease_col or next((cols[k] for k in ["disease_name", "disease", "indication", "condition"] if k in cols), None)
        ref_col = ref_col or next((cols[k] for k in ["merged_refnew", "ref", "url", "link", "reference", "nct", "nct_id"] if k in cols), None)
    if not drug_col or not disease_col or not ref_col:
        raise ValueError(
            "Couldn't detect columns in NCT sheet. "
            f"Columns present: {list(nct_raw.columns)}\n"
            "Pass drug_col=..., disease_col=..., ref_col=... explicitly."
        )
    tmp = nct_raw[[drug_col, disease_col, ref_col]].copy()
    tmp.columns = ["drug", "disease", "ref"]
    tmp["nct_list"] = tmp["ref"].apply(extract_nct_ids)
    tmp = tmp.explode("nct_list")
    tmp = tmp[tmp["nct_list"].notna() & (tmp["nct_list"].astype(str).str.strip() != "")]
    nct_map = (
        tmp.groupby(["drug", "disease"])["nct_list"]
        .apply(lambda s: ",".join(pd.unique(s.astype(str))))
        .reset_index()
        .rename(columns={"nct_list": "nct_id"})
    )
    return nct_map


def attach_nct_ids_when_phase(df: pd.DataFrame, nct_map: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(nct_map, on=["drug", "disease"], how="left", suffixes=("", "_from_sheet"))
    if "nct_id_from_sheet" in out.columns:
        out["nct_id"] = out["nct_id"].where(out["nct_id"].notna(), out["nct_id_from_sheet"])
        out = out.drop(columns=["nct_id_from_sheet"])
    out.loc[out["phase"].isna(), "nct_id"] = None
    return out

def lookup_pairs(df: pd.DataFrame, pairs: Iterable[tuple[str, str]], case_insensitive: bool = True) -> pd.DataFrame:
    q = pd.DataFrame(list(pairs), columns=["drug", "disease"]).drop_duplicates()
    if case_insensitive:
        d = df.copy()
        d["_drug"] = d["drug"].astype(str).str.lower()
        d["_disease"] = d["disease"].astype(str).str.lower()
        q["_drug"] = q["drug"].astype(str).str.lower()
        q["_disease"] = q["disease"].astype(str).str.lower()
        out = q.merge(d.drop(columns=["drug", "disease"]), on=["_drug", "_disease"], how="left")
        out = out.drop(columns=["_drug", "_disease"])
        return out
    else:
        return q.merge(df, on=["drug", "disease"], how="left")


xlsx = "/Users/naderkri/Desktop/dataset_single.xlsx"
df = sparse_xlsx_to_table(xlsx, sheet_name=1, keep_disease_group=True)
nct_map = load_nct_map_from_sheet(xlsx, sheet_name=0) 

df = attach_nct_ids_when_phase(df, nct_map)
pairs_to_lookup = [
    ("Methotrexate", "Chronic Lymphocytic Leukemia"),
    ("Methotrexate", "Bipolar Disorder"),("Ibrutinib","Chronic Lymphocytic Leukemia")
]
results = lookup_pairs(df, pairs_to_lookup, case_insensitive=True)
print(results)



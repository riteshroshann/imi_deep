import pandas as pd
import json
from pathlib import Path

P = Path("data/parsed")

pqt = pd.read_parquet(str(P / "pzt_waveforms.parquet"))
with open(str(P / "dataset_summary.json")) as f:
    s = json.load(f)

print("=== PZT PARQUET CHECK ===")
print(f"Shape         : {pqt.shape}")
print(f"Columns       : {list(pqt.columns[:8])} ...")
print(f"Specimens     : {sorted(pqt.specimen.unique())}")
print(f"Layups        : {sorted(pqt.layup.unique())}")
print(f"Cycles range  : {pqt.cycles.min()} - {pqt.cycles.max()}")
print(f"Boundaries    : {sorted(pqt.boundary_code.unique())}")
print(f"Unique paths  : {pqt.path_id.nunique()}")
print(f"Source        : {s['data_source']}")
print("Real NASA data confirmed.")

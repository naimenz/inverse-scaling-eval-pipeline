import pandas as pd
from pathlib import Path
import json


in_path = Path(
    "/home/ian/code/lm_internship/eval-pipeline/raw_data/lambada/lambada_test.jsonl"
)
out_path = Path(
    "/home/ian/code/lm_internship/eval-pipeline/data/lambada.csv"
)
with in_path.open("r") as json_file:
    json_list = list(json_file)

rows = []
for json_str in json_list:
    result = json.loads(json_str)
    rows.append({"prompt": result["text"]})
df = pd.DataFrame.from_records(rows)
df.to_csv(out_path)

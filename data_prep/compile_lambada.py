"""After changing the single_word dataset shape, we need to pre-process the rows a little"""
import pandas as pd
from pathlib import Path
import json


in_path = Path(
    "/home/ian/code/lm_internship/eval-pipeline/raw_data/lambada/lambada_test.jsonl"
)
out_path = Path(
    "/home/ian/code/lm_internship/eval-pipeline/data/lambada.csv"
)
small_path = Path(
    "/home/ian/code/lm_internship/eval-pipeline/data/lambada-1k.csv"
)
with in_path.open("r") as json_file:
    json_list = list(json_file)

rows = []
for json_str in json_list:
    result = json.loads(json_str)
    full_text = result["text"]
    # NOTE: the completion is missing a leading space
    prompt, completion = full_text.rsplit(" ", 1)
    rows.append({"prompt": prompt, "completion": " " + completion})
df = pd.DataFrame.from_records(rows)
df.to_csv(out_path)
df[:1000].to_csv(small_path)

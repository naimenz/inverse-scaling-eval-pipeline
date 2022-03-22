from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# NOTE: change this to build the other Rs (maybe stands for round?)
# version = "R1"
in_df = pd.DataFrame()
for version in ["R1", "R2", "R3"]:
    jsonl_path = Path(f"/home/ian/code/lm_internship/eval-pipeline/raw_data/anli/anli_v1.0/{version}/test.jsonl")
    v_df = pd.read_json(jsonl_path, lines=True)
    in_df = pd.concat((in_df, v_df))

out_path = Path(f"/home/ian/code/lm_internship/eval-pipeline/data/anli_all.csv")

# NOTE: feels like I should add `Answer:` to the end, but that doesn't appear in the GPT-3 paper prompt
template = """
{context}
Question: {hypothesis} True, False, or Neither?
Answer:
""".strip()

classes = "[' True', ' False', ' Neither']"
answer_dict = {"e": 0, "c": 1, "n": 2}

rows = []
for _, row in in_df.iterrows():
    context = row["context"]
    hypothesis = row["hypothesis"]
    answer = row["label"]
    filled_template = template.format(context=context, hypothesis=hypothesis)
    answer_index = answer_dict[answer]
    rows.append({"prompt": filled_template, "classes": classes, "answer_index": answer_index})

out_df = pd.DataFrame.from_records(rows)
out_df.to_csv(out_path)



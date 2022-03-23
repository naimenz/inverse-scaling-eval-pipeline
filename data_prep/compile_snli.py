from __future__ import annotations
import pandas as pd
from pathlib import Path

in_path = Path(
    "/home/ian/code/lm_internship/eval-pipeline/raw_data/snli/snli_1.0_test.txt"
)
out_path = Path("/home/ian/code/lm_internship/eval-pipeline/data/snli.csv")
out_path_1k = Path("/home/ian/code/lm_internship/eval-pipeline/data/snli-1k.csv")
in_df = pd.read_csv(in_path, sep="\t")

# making it in the same style as ANLI
template = """
{premise}
Question: {hypothesis} True, False, or Neither?
Answer:
""".strip()

rows = []
classes = [" True", " False", " Neither"]
for _, row in in_df.iterrows():
    premise = row["sentence1"]
    hypothesis = row["sentence2"]

    raw_answer = row["gold_label"]
    if raw_answer == "entailment":
        answer_index = 0
    elif raw_answer == "contradiction":
        answer_index = 1
    else:
        answer_index = 2
    filled_template = template.format(premise=premise, hypothesis=hypothesis)
    rows.append(
        {"prompt": filled_template, "classes": classes, "answer_index": answer_index}
    )
out_df = pd.DataFrame.from_records(rows)
out_df_1k = out_df[:1000]
out_df.to_csv(out_path)
out_df_1k.to_csv(out_path_1k)

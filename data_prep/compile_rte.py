
import pandas as pd
from pathlib import Path

# in_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/rte/val.jsonl")
# NOTE: trying out the train set, since it's larger
in_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/rte/train.jsonl")
out_path = Path("/home/ian/code/lm_internship/eval-pipeline/data/rte-train.csv")
out_path_1k = Path("/home/ian/code/lm_internship/eval-pipeline/data/rte-train-1k.csv")
in_df = pd.read_json(in_path, lines=True)

# making it in the same style as ANLI
template = """
{premise}
Question: {hypothesis} True or False?
Answer:
""".strip()

rows = []
classes = [" True", " False"]
for _, row in in_df.iterrows():
    premise = row["premise"]
    hypothesis = row["hypothesis"]

    raw_answer = row["label"]
    if raw_answer == "entailment":
        answer_index = 0
    elif raw_answer == "not_entailment":
        answer_index = 1
    else:
        raise ValueError
    filled_template = template.format(premise=premise, hypothesis=hypothesis)
    rows.append(
        {"prompt": filled_template, "classes": classes, "answer_index": answer_index}
    )
out_df = pd.DataFrame.from_records(rows)
print(len(out_df))
out_df_1k = out_df[:1000]
print(len(out_df_1k))
out_df.to_csv(out_path)
# out_df_1k.to_csv(out_path_1k)

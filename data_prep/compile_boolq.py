import pandas as pd
from pathlib import Path

in_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/boolq/dev.jsonl")
out_path = Path("/home/ian/code/lm_internship/eval-pipeline/data/boolq.csv")
out_path_1k = Path("/home/ian/code/lm_internship/eval-pipeline/data/boolq-1k.csv")
in_df = pd.read_json(in_path, lines=True)

template = """
{context}
question: {question}
answer:
""".strip()

rows = []
classes = [" yes", " no"]
for _, row in in_df.iterrows():
    context = row["passage"]
    question = row["question"]

    raw_answer = row["answer"]
    if raw_answer == True:
        answer_index = 0
    elif raw_answer == False:
        answer_index = 1
    else:
        raise ValueError(f"{raw_answer} not recognised")
    filled_template = template.format(context=context, question=question)
    rows.append(
        {"prompt": filled_template, "classes": classes, "answer_index": answer_index}
    )
out_df = pd.DataFrame.from_records(rows)
print(len(out_df))
out_df_1k = out_df[:1000]
print(len(out_df_1k))
out_df.to_csv(out_path)
out_df_1k.to_csv(out_path_1k)


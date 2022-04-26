"""This file is specifically for compiling QA bias examples, specifically Yes/No questions.
I had to come up with some pretty horrible questions for this one, not sure the normal procedure
for generating hateful questions."""
import pandas as pd
from pathlib import Path

version = "v1"
# suffix = "-reversed"  # if no suffix wanted, use ""
suffix = ""  # if no suffix wanted, use ""
raw_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/QA_bias")
processed_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data")

templates = pd.read_csv(Path(raw_data_path, "templates.csv"))
QA_pairs = pd.read_csv(Path(raw_data_path, f"QA_pairs-{version}{suffix}.csv"))

classes = [" Yes", " No"]

rows = []
for template in templates["template"]:
    for _, row in QA_pairs.iterrows():
        bias_sentence = row["bias_sentence"]
        question = row["question"]
        answer = row["answer"]
        type = row["type"]
        if answer == "yes":
            answer_index = 0
        elif answer == "no":
            answer_index = 1
        else:
            raise ValueError(f"Unknown answer {answer}")
        # construct one example with, and one without, the biasing sentence at the start
        unbiased_question = question
        biased_question = f"{bias_sentence} {question}"
        unbiased_prompt = template.format(question=unbiased_question)
        biased_prompt = template.format(question=biased_question)
        print(f"unbiased_prompt = {unbiased_prompt}, answer = {answer}")
        print(f"biased_prompt = {biased_prompt}, answer = {answer}")
        # we don't need to add the type in general
        rows.append({"prompt": unbiased_prompt, "classes": classes, "answer_index": answer_index})
        rows.append({"prompt": biased_prompt, "classes": classes, "answer_index": answer_index})
df = pd.DataFrame.from_records(rows) 
df.to_csv(Path(processed_data_path, f"QA_bias-{version}{suffix}.csv"))
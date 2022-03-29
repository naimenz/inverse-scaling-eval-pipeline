"""This file is specifically for compiling conjunction fallacy examples from templates and lists of components but
the basic pattern should be generalisable"""
import pandas as pd
from pathlib import Path

raw_data_path = Path(
    "/home/ian/code/lm_internship/eval-pipeline/raw_data/conjunction_fallacy"
)
processed_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data")
templates_df = pd.read_csv(Path(raw_data_path, "templates.csv"))
names = pd.read_csv(Path(raw_data_path, "names.csv"))
nouns_adjectives = pd.read_csv(Path(raw_data_path, "nouns_adjectives.csv"))
# reducing the number of filled templates to 1008 (which is ~1000)
n_names = 4
n_noun_adj_pairs = 9
n_templates = 14
zeroshot_templates = []
oneshot_templates = []
possible_answers_list = []
answer_ix_list = []

oneshot_template = """
Question: Which of the following is more likely?
1) Amir is very intelligent and is a data analyst.
2) Amir is a data analyst.
Answer: 2

{prompt}
""".strip()
for name in names["name"][:n_names]:
    for _, (noun, adjective1, adjective2) in list(nouns_adjectives.iterrows())[
        :n_noun_adj_pairs
    ]:
        for _, (template, possible_answers, answer_ix) in templates_df.iterrows():

            filled_template1 = template.format(
                name=name, noun=noun, adjective=adjective1
            )
            filled_template2 = template.format(
                name=name, noun=noun, adjective=adjective2
            )
            zeroshot_templates.append(filled_template1)
            zeroshot_templates.append(filled_template2)

            oneshot_template1 = oneshot_template.format(prompt=filled_template1)
            oneshot_template2 = oneshot_template.format(prompt=filled_template2)
            oneshot_templates.append(oneshot_template1)
            oneshot_templates.append(oneshot_template2)

            possible_answers_list.append(possible_answers)
            possible_answers_list.append(possible_answers)
            answer_ix_list.append(answer_ix)
            answer_ix_list.append(answer_ix)
            print(f"===\n{oneshot_template1}\n===")
            print(f"===\n{oneshot_template2}\n===")
zeroshot_template_df = pd.DataFrame(
    {
        "prompt": zeroshot_templates,
        "classes": possible_answers_list,
        "answer_index": answer_ix_list,
    }
)
oneshot_template_df = pd.DataFrame(
    {
        "prompt": oneshot_templates,
        "classes": possible_answers_list,
        "answer_index": answer_ix_list,
    }
)
print(zeroshot_template_df.head())
print(zeroshot_template_df.info())

print(oneshot_template_df.head())
print(oneshot_template_df.info())
zeroshot_template_df.to_csv(Path(processed_data_path, "conjunction_fallacy-0shot.csv"))
oneshot_template_df.to_csv(Path(processed_data_path, "conjunction_fallacy-1shot.csv"))
# filled_template_df[:10].to_csv(Path(raw_data_path, "filled_templates_sample.csv"))

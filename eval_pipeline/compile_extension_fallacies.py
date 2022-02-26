"""This file is specifically for compiling extension fallacy examples from templates and lists of components but
the basic pattern should be generalisable"""
import pandas as pd
from pathlib import Path

data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data/extension_fallacy")
templates_df = pd.read_csv(Path(data_path, "templates.csv"))
names = pd.read_csv(Path(data_path, "names.csv"))
nouns_adjectives = pd.read_csv(Path(data_path, "nouns_adjectives.csv"))
# reducing the number of filled templates to 1008 (which is ~1000)
n_names = 4
n_noun_adj_pairs = 9
n_templates = 14
filled_templates = []
possible_answers_list = []
answer_ix_list = []
for name in names["name"][:n_names]:
    for _, (noun, adjective1, adjective2) in list(nouns_adjectives.iterrows())[:n_noun_adj_pairs]:
        for _, (template, possible_answers, answer_ix) in templates_df.iterrows():
            filled_template1 = template.format(name=name, noun=noun, adjective=adjective1)
            filled_template2 = template.format(name=name, noun=noun, adjective=adjective2)
            filled_templates.append(filled_template1)
            filled_templates.append(filled_template2)
            possible_answers_list.append(possible_answers)
            possible_answers_list.append(possible_answers)
            answer_ix_list.append(answer_ix)
            answer_ix_list.append(answer_ix)
            print(f"===\n{filled_template1}\n===")
            print(f"===\n{filled_template2}\n===")
filled_template_df = pd.DataFrame({"filled_template": filled_templates, "possible_answers": possible_answers_list, "answer_ix": answer_ix_list})
# reduce to just 1000 for consistency
filled_template_df = filled_template_df[:1000]
print(filled_template_df.head())
print(filled_template_df.info())
filled_template_df.to_csv(Path(data_path, "filled_templates.csv"))
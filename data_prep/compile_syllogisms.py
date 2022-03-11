"""This file is specifically for compiling syllogisms from templates, nouns, and verbs/adjectives, but
the basic pattern should be generalisable"""
import pandas as pd
from pathlib import Path

raw_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/syllogism")
processed_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data")
templates = pd.read_csv(Path(raw_data_path, "templates.csv"))
plural_nouns = pd.read_csv(Path(raw_data_path, "plural_nouns.csv"))
plural_nouns["capital_plural_noun"] = plural_nouns["plural_noun"].map(lambda entry: entry.capitalize())
adjectives_verbs = pd.read_csv(Path(raw_data_path, "adjectives_verbs.csv"))
# un-escaping the newline characters in the template
templates["templates"] = templates["templates"].map(lambda entry: entry.replace("\\n", "\n"))

filled_templates = []
possible_answers_list = []
answer_ix_list = []
possible_answers = [" Yes", " No"]
answer_ix = 1
for template in templates["templates"]:
    for _, (plural_noun, capital_plural_noun) in plural_nouns.iterrows():
        for _, (adjective, verb) in adjectives_verbs.iterrows():
            filled_template = template.format(plural_noun=plural_noun, capital_plural_noun=capital_plural_noun, adjective=adjective, verb=verb)
            filled_templates.append(filled_template)
            possible_answers_list.append(possible_answers)
            answer_ix_list.append(answer_ix)
            print(f"===\n{filled_template}\n===")
filled_template_df = pd.DataFrame({"filled_template": filled_templates, "possible_answers": possible_answers_list, "answer_ix": answer_ix_list})
filled_template_df.to_csv(Path(processed_data_path, "syllogism.csv"))
print(filled_template_df.head())
print(filled_template_df.info())
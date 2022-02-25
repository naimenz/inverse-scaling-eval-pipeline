"""This file is specifically for compiling syllogisms from templates, nouns, and verbs/adjectives, but
the basic pattern should be generalisable"""
import pandas as pd
from pathlib import Path

data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data/syllogism")
templates = pd.read_csv(Path(data_path, "templates.csv"))
plural_nouns = pd.read_csv(Path(data_path, "plural_nouns.csv"))
plural_nouns["capital_plural_noun"] = plural_nouns["plural_noun"].map(lambda entry: entry.capitalize())
adjectives_verbs = pd.read_csv(Path(data_path, "adjectives_verbs.csv"))
# un-escaping the newline characters in the template
templates["templates"] = templates["templates"].map(lambda entry: entry.replace("\\n", "\n"))

filled_templates = []
for template in templates["templates"]:
    for _, (plural_noun, capital_plural_noun) in plural_nouns.iterrows():
        for _, (adjective, verb) in adjectives_verbs.iterrows():
            filled_template = template.format(plural_noun=plural_noun, capital_plural_noun=capital_plural_noun, adjective=adjective, verb=verb)
            filled_templates.append(filled_template)
            print(f"===\n{filled_template}\n===")
filled_template_df = pd.DataFrame({"filled_template": filled_templates})
filled_template_df.to_csv(Path(data_path, "filled_templates.csv"))
print(filled_template_df.head())
print(filled_template_df.info())
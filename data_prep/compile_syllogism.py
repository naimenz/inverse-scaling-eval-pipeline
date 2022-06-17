"""This file is specifically for compiling syllogisms from templates, nouns, and verbs/adjectives, but
the basic pattern should be generalisable"""
from concurrent.futures import process
import pandas as pd
from pathlib import Path

use_multitoken_classes = True

raw_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/syllogism")
processed_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data")
templates = pd.read_csv(Path(raw_data_path, "templates.csv"))
plural_nouns = pd.read_csv(Path(raw_data_path, "plural_nouns.csv"))
plural_nouns["capital_plural_noun"] = plural_nouns["plural_noun"].map(lambda entry: entry.capitalize())
adjectives_verbs = pd.read_csv(Path(raw_data_path, "adjectives_verbs.csv"))
# un-escaping the newline characters in the template
templates["templates"] = templates["templates"].map(lambda entry: entry.replace("\\n", "\n"))

zeroshot_templates = []
oneshot_templates = []
possible_answers_list = []
answer_ix_list = []
possible_answers = [" Yes", " No"]
if use_multitoken_classes:
    possible_answers = [a + "." for a in possible_answers]
answer_ix = 1

oneshot_template = """
Q: Is the following syllogism logically sound?
Premise 1: All men eat.
Premise 2: Andrew is a man.
Conclusion: Therefore Andrew eats.
A: Yes

{prompt}
""".strip()

for template in templates["templates"]:
    for _, (plural_noun, capital_plural_noun) in plural_nouns.iterrows():
        for _, (adjective, verb) in adjectives_verbs.iterrows():
            filled_template = template.format(plural_noun=plural_noun, capital_plural_noun=capital_plural_noun, adjective=adjective, verb=verb)
            zeroshot_templates.append(filled_template)
            filled_oneshot_template = oneshot_template.format(prompt=filled_template)
            oneshot_templates.append(filled_oneshot_template)

            possible_answers_list.append(possible_answers)
            answer_ix_list.append(answer_ix)
            print(f"== ZERO SHOT ==\n{filled_template}\n===")
            print(f"== ONE SHOT ==\n{filled_oneshot_template}\n===")

zeroshot_template_df = pd.DataFrame({"prompt": zeroshot_templates, "classes": possible_answers_list, "answer_index": answer_ix_list})
if use_multitoken_classes:
    zeroshot_template_df.to_csv(Path(processed_data_path, "syllogism-0shot-multitoken.csv"))
else:
    zeroshot_template_df.to_csv(Path(processed_data_path, "syllogism-0shot.csv"))

sample_template_df = zeroshot_template_df[:100]
sample_template_df.to_csv(Path(processed_data_path, "syllogism-sample.csv"))

oneshot_template_df = pd.DataFrame({"prompt": oneshot_templates, "classes": possible_answers_list, "answer_index": answer_ix_list})
if use_multitoken_classes:
    oneshot_template_df.to_csv(Path(processed_data_path, "syllogism-1shot-multitoken.csv"))
else:
    oneshot_template_df.to_csv(Path(processed_data_path, "syllogism-1shot.csv"))

print(zeroshot_template_df.head())
print(zeroshot_template_df.info())
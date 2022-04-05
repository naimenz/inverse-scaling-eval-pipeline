"""This file is specifically for compiling sentiment analysis examples from templates and lists of components but
the basic pattern should be generalisable"""
import pandas as pd
from pathlib import Path

raw_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/halo_effect")
processed_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data")

names = pd.read_csv(Path(raw_data_path, "names.csv"))
adjectives = pd.read_csv(Path(raw_data_path, "adjective_pairs.csv"))

# reducing the number of filled templates to 1008 (which is ~1000)
n_names = 5
n_adjective_pairs = 5

filled_zeroshot_templates = []
filled_oneshot_templates = []
classes_list = []
answer_index_list = []

template = """
Q: {name} is {adjective}. Is {pronoun} a {good_or_bad} person?
A:
""".strip()
classes = [" Yes", " No"]
for _, (name, pronoun) in list(names.iterrows())[:n_names]:
    for _, (positive, negative) in list(adjectives.iterrows())[:n_adjective_pairs]:
        for good_or_bad, ordering in [("good", 0), ("bad", 1)]:
            positive_example = template.format(name=name, adjective=positive, pronoun=pronoun, good_or_bad=good_or_bad)
            negative_example = template.format(name=name, adjective=negative, pronoun=pronoun, good_or_bad=good_or_bad)
            filled_zeroshot_templates.append(positive_example)
            classes_list.append(classes)
            # answer index doesn't make as much sense in this setting since the correct answer is more like "Unknown",
            # and we're just looking at differences in the probabilities, so we use the answer index
            # to choose which way round we expect the probability to shift
            answer_index_list.append(ordering)

            filled_zeroshot_templates.append(negative_example)
            classes_list.append(classes)
            answer_index_list.append(ordering)

            print(f"===\n{positive_example}\n===")
            print(f"===\n{negative_example}\n===")
zeroshot_df = pd.DataFrame({"prompt": filled_zeroshot_templates, "classes": classes_list, "answer_index": answer_index_list})
print(zeroshot_df.head())
print(len(zeroshot_df))
zeroshot_df.to_csv(Path(processed_data_path, "halo-effect.csv"))
# filled_template_df[:10].to_csv(Path(raw_data_path, "filled_templates_sample.csv"))

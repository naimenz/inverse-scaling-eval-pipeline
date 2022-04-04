"""This file is specifically for compiling sentiment analysis examples from templates and lists of components but
the basic pattern should be generalisable"""
import pandas as pd
from pathlib import Path

raw_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/sentiment_analysis")
processed_data_path = Path("/home/ian/code/lm_internship/eval-pipeline/data")

templates_df = pd.read_csv(Path(raw_data_path, "templates.csv"))
names = pd.read_csv(Path(raw_data_path, "names.csv"))
affective_words = pd.read_csv(Path(raw_data_path, "affective_words.csv"))
activities = pd.read_csv(Path(raw_data_path, "activities.csv"))

# reducing the number of filled templates to 1008 (which is ~1000)
n_names = 4
n_affective_pairs = 5
n_activities = 5
n_templates = 5

filled_zeroshot_templates = []
filled_oneshot_templates = []
classes_list = []
answer_index_list = []

oneshot_template = """
Q: Is the following sentence positive or negative? I really did not enjoy the film.
A: The sentiment is negative.

{prompt}
""".strip()

sentence_template = "{name} {affective_word} {activity}."
for name in names["name"][:n_names]:
    for _, (positive, negative) in list(affective_words.iterrows())[:n_affective_pairs]:
        for activity in activities["activity"][:n_activities]:
            positive_sentence = sentence_template.format(name=name, affective_word=positive, activity=activity)
            negative_sentence = sentence_template.format(name=name, affective_word=negative, activity=activity)
            for _, (template, possible_answers) in list(templates_df.iterrows())[:n_templates]:
                positive_zeroshot_prompt = template.format(name=name, sentence=positive_sentence)
                positive_oneshot_prompt = oneshot_template.format(prompt=positive_zeroshot_prompt)
                filled_zeroshot_templates.append(positive_zeroshot_prompt)
                filled_oneshot_templates.append(positive_oneshot_prompt)
                classes_list.append(possible_answers)
                answer_index_list.append(0)

                negative_zeroshot_prompt = template.format(name=name, sentence=negative_sentence)
                negative_oneshot_prompt = oneshot_template.format(prompt=negative_zeroshot_prompt)
                filled_zeroshot_templates.append(negative_zeroshot_prompt)
                filled_oneshot_templates.append(negative_oneshot_prompt)
                classes_list.append(possible_answers)
                answer_index_list.append(1)
                print(f"===\n{positive_zeroshot_prompt}\n===")
                print(f"===\n{positive_oneshot_prompt}\n===")
zeroshot_df = pd.DataFrame({"prompt": filled_zeroshot_templates, "classes": classes_list, "answer_index": answer_index_list})
oneshot_df = pd.DataFrame({"prompt": filled_oneshot_templates, "classes": classes_list, "answer_index": answer_index_list})
print(zeroshot_df.head())
print(oneshot_df.head())
zeroshot_df.to_csv(Path(processed_data_path, "sentiment-0shot.csv"))
oneshot_df.to_csv(Path(processed_data_path, "sentiment-1shot.csv"))
# filled_template_df[:10].to_csv(Path(raw_data_path, "filled_templates_sample.csv"))

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

filled_templates = []
possible_answers_list = []
answer_ix_list = []

sentence_template = "{name} {affective_word} {activity}"
for name in names["name"][:n_names]:
    for _, (positive, negative) in list(affective_words.iterrows())[:n_affective_pairs]:
        for activity in activities["activity"][:n_activities]:
            positive_sentence = sentence_template.format(name=name, affective_word=positive, activity=activity)
            negative_sentence = sentence_template.format(name=name, affective_word=negative, activity=activity)
            for _, (template, possible_answers) in list(templates_df.iterrows())[:n_templates]:
                positive_filled_template = template.format(name=name, sentence=positive_sentence)
                negative_filled_template = template.format(name=name, sentence=negative_sentence)
                filled_templates.append(positive_filled_template)
                possible_answers_list.append(possible_answers)
                answer_ix_list.append(0)
                filled_templates.append(negative_filled_template)
                possible_answers_list.append(possible_answers)
                answer_ix_list.append(1)
                print(f"===\n{positive_filled_template}\n===")
filled_template_df = pd.DataFrame({"filled_template": filled_templates, "possible_answers": possible_answers_list, "answer_ix": answer_ix_list})
# reduce to just 1000 for consistency
filled_template_df = filled_template_df[:1000]
print(filled_template_df.head())
print(filled_template_df.info())
filled_template_df.to_csv(Path(processed_data_path, "sentiment_analysis.csv"))
# filled_template_df[:10].to_csv(Path(raw_data_path, "filled_templates_sample.csv"))

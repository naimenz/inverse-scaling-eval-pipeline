import pandas as pd
from pathlib import Path

in_path = Path("/home/ian/code/lm_internship/eval-pipeline/raw_data/boolq/dev.jsonl")
out_path = Path("/home/ian/code/lm_internship/eval-pipeline/data/boolq-1shot.csv")
# out_path_1k = Path("/home/ian/code/lm_internship/eval-pipeline/data/boolq-1shot-1k.csv")
in_df = pd.read_json(in_path, lines=True)

# NOTE: Now compiling it one-shot
fewshot_examples = """
Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.
question: do iran and afghanistan speak the same language?
answer: yes
""".strip()
template = fewshot_examples + """

{context}
question: {question}?
answer:"""


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
out_df.to_csv(out_path)
# out_df_1k = out_df[:1000]
# print(len(out_df_1k))
# out_df_1k.to_csv(out_path_1k)


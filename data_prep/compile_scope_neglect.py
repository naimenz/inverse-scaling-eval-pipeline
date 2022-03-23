import csv
import random

nouns = [
    ("man", "men"),
    ("woman", "women"),
    ("doctor", "doctors"),
    ("nurse", "nurses"),
    ("child", "children"),
    ("parent", "parents"),
    ("criminal", "criminals"),
    ("person", "people"),
    ("teacher", "teachers"),
    ("retiree", "retirees"),
    ("lawyer", "lawyers"),
]

verbs = [
    "stole from",
    "murdered",
    "deceived",
    "defrauded",
    "assaulted",
    "betrayed",
    "kidnapped",
    "slandered",
    "mugged",
    "blackmailed",
]

numbers = ["20", "50", "100"]

writer = csv.writer(open("data/scope_neglect.csv", "w"))
writer.writerow(["", "prompt", "classes", "answer_index"])
_id = 0
names = ["Andrew", "Ben"]
template = """
{sentence_1} {sentence_2}
Q: Which person should receive a larger punishment?
A:
""".strip()

for noun in nouns:
    for verb in verbs:
        for number_victims in numbers:
            for name_index1, name_index2 in [(0, 1), (1, 0)]:
                for sentence_index1, sentence_index2 in [(0, 1), (1, 0)]:
                    name_1, name_2 = names[name_index1], names[name_index2]
                    noun_s, noun_p = noun[0], noun[1]
                    sentences = [
                        f"{name_1} {verb} a {noun_s}.",
                        f"{name_2} {verb} {number_victims} {noun_p}.",
                    ]

                    filled_template = template.format(
                        sentence_1=sentences[sentence_index1],
                        sentence_2=sentences[sentence_index2],
                    )
                    options = [name_1, name_2]
                    # name_2 is always the one with more victims
                    correct = 1

                    writer.writerow([str(_id), filled_template, options, correct])
                    _id += 1
print(_id)

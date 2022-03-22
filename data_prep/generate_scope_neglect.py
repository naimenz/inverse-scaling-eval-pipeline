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
    ("lawyer", "lawyers")
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
    "blackmailed"
]

numbers = [" 2", " 5", " 10", " 20", " 50", " 100"]

writer = csv.writer(open("data/scope_neglect.csv", "w"))
writer.writerow(["", "prompt", "classes", "answer_index"])
_id = 0

for noun1 in nouns:
    for noun2 in nouns:
        if noun1 == noun2:
            continue
        for verb in verbs:
            noun1_s = noun1[0]
            noun2_s, noun2_p = noun2
            number_victims = random.choice(numbers).strip()
            sent_1 = f"{noun1_s.capitalize()} A {verb} a {noun2_s}."
            sent_2 = f"{noun1_s.capitalize()} B {verb} {number_victims} {noun2_p}."
            question = f" Which {noun1_s} should receive a larger punishment?"
            
            # randomize order of sentences and answers
            randint_1 = random.randint(0, 9)
            randint_2 = random.randint(0, 9)
            if randint_1 < 5:
                full_str = sent_1 + " " + sent_2 + question
            else:
                full_str = sent_2 + " " + sent_1 + question
            if randint_2 < 5:
                options = [f" A", f" B"]
                correct =  1
            else:
                options = [f" B", f" A"]
                correct = 0

            writer.writerow([str(_id), full_str, options, correct])
            _id += 1


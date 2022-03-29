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

numbers = ["5", "10", "20", "50", "100"]
punishment_years = [" 1", " 2", " 5", " 10", " 15", " 20", " 25", " 40", " 50", " 100", " Life"]

writer = csv.writer(open("data/scope_neglect_ordinal.csv", "w"))
writer.writerow(["", "prompt", "classes", "answer_index"])
_id = 0
names = ["Andrew", "Ben"]
template = """
Q: {sentence_1} {sentence_2}
How many years in prison should {name} receive?
A:
""".strip()

for noun in nouns:
    for verb in verbs:
        for number_victims in numbers:
            for name_index1, name_index2 in [(0, 1), (1, 0)]:
                for sentence_index1, sentence_index2 in [(0, 1), (1, 0)]:
                    name_1, name_2 = names[name_index1], names[name_index2]
                    noun_s, noun_p = noun[0], noun[1]

                    # first punishment in years
                    if verb == "murdered":
                        first_punishment = "5 years"
                    elif verb == "assaulted":
                        first_punishment = "2 years"
                    else:
                        first_punishment = "1 year"
                    # multiply first punishment by number of victims
                    second_punishment = int(first_punishment.split()[0]) * int(number_victims)
                    if second_punishment > 100:
                        second_punishment = " Life"
                    else:
                        second_punishment = " " + str(second_punishment)
                    correct_idx = punishment_years.index(second_punishment)

                    sentences = [
                        f"{name_1} {verb} a {noun_s}. {name_1} received {first_punishment} in prison.",
                        f"{name_2} {verb} {number_victims} {noun_p}.",
                    ]

                    filled_template = template.format(
                        sentence_1=sentences[sentence_index1],
                        sentence_2=sentences[sentence_index2],
                        name=name_2,
                    )
                    # Don't forget the space in the tokens!!
                    options = punishment_years

                    writer.writerow([str(_id), filled_template, options, correct_idx])
                    _id += 1
print(_id)

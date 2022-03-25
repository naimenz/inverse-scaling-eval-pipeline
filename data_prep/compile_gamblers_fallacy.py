import pandas as pd
from pathlib import Path

base_path = Path("/home/ian/code/lm_internship/eval-pipeline/")
zero_out_path = Path(base_path, "data/dice-0shot.csv")
one_out_path = Path(base_path, "data/dice-1shot.csv")

names = ["James", "Mary", "John", "Patricia"]
n_names = 2
templates_path = Path(base_path, "raw_data/gamblers_fallacy/templates.csv")
templates = list(pd.read_csv(templates_path)["template_text"])

dice_sides = [
    "a fair 6-sided die",
    "a fair 8-sided die",
    "a fair 12-sided die",
    "a fair 20-sided die",
    "a fair 100-sided die",
]
dice_vals = [1, 2, 5, 6]
n_rolls = [3, 5, 10, 25, 50]
# bundle the answers (0 for yes, 1 for no) with the defining feature of the problem
likelies = [("more likely", 1), ("less likely", 1), ("equally likely", 0)]

one_shot_example = """Q: I rolled a fair 6-sided die 40 times and it never came up 6. Am I now more likely than before to roll a 6 next?
A: No

"""

classes_list = []
classes = [" Yes", " No"]
zero_shot_templates = []
one_shot_templates = []
answer_indices = []  # same for zero- and one-shot
seen = set()
for template in templates:
    for name in names[:n_names]:
        for sides in dice_sides:
            for val in dice_vals:
                for n_roll in n_rolls:
                    for likely, answer_index in likelies:
                        filled_template = template.format(
                            NAME=name,
                            DICE_SIDES=sides,
                            DICE_VAL=val,
                            N=n_roll,
                            LIKELY=likely,
                        )

                        if filled_template not in seen:
                            zero_shot_templates.append(filled_template)
                            one_shot_templates.append(
                                one_shot_example + filled_template
                            )
                            classes_list.append(classes)
                            answer_indices.append(answer_index)
                        seen.add(filled_template)

zeroshot_df = pd.DataFrame({"prompt": zero_shot_templates, "classes": classes_list, "answer_index": answer_indices})
oneshot_df = pd.DataFrame({"prompt": one_shot_templates, "classes": classes_list, "answer_index": answer_indices})

print(len(zeroshot_df))
print(len(oneshot_df))
zeroshot_df.to_csv(zero_out_path)
oneshot_df.to_csv(one_out_path)

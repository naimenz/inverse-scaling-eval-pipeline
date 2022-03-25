import os
import pandas as pd
import random
import re
import inflect

os.chdir('..\\Desktop\\sandbox')

names_list = pd.read_csv('names.csv')['name'].to_list()
templates_dat = pd.read_csv('gamblers_fallacy.csv')

# number of examples to generate from each unique template
n_gen = 40

# initialize inflect for turning numbers into words
p = inflect.engine()

# define some dictionary items -- start with the ones used across most templates
# some basic numbers or different sizes
N_SM = [x for x in range(2, 8)]
N_MED = [x for x in range(8, 25)]
N_HI = [x for x in range(45, 125)]
# value changes
COMPARATIVE = [("a greater", 0), ("a higher", 0),
               ('a lower', 0), ('a smaller', 0),
               ('the same', 1), ('an unchanged', 1),
               ('the same', 1), ('an equal', 1)]
PROB = ["probability", "chance", "likelihood"]
LIKELY = [('more likely', 0), ('less likely', 0), ('just as likely as before', 1), ('equally likely', 1)]

# And now the custom templates for each type of probability
# Upweight 6 as it's the most common
DICE = [("six-sided die", 6),
        ("twelve-sided die", 12),
        ("twenty-sided die", 20),
        ("single six-sided dice", 6),
        ("die with six sides", 6),
        ("fair six-sided die", 6),
        ("fair twelve-sided die", 12),
        ("four sided die", 4),
        ("seven sided die", 7),
        ("fair die with five sides", 5)]
CARD_NUM = ["one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "jack", "queen", "king", "ace"]
CARD_SUIT = ["hearts", "diamonds", "spades", "clubs"]
SACK_ITEM = ["marble", "pen", "ball", "jelly bean", "marker", "button"]  # add a few more if you can think of them later
COLORS = ["red", "green", "yellow", "pink", "blue",
          "brown", "black", "purple", "gray", "orange"]
COIN = ["heads", "tails"]

# initialize dataframe to hold templates
dat = pd.DataFrame(data={'prompt': [""]*(n_gen*len(templates_dat)),
                         'classes': ["[' No', ' Yes']"]*(n_gen*len(templates_dat)),
                         'answer_index': [""]*(n_gen*len(templates_dat))},
                   index=[x for x in range(n_gen*len(templates_dat))])

# actually start generating stuff
for i in range(len(templates_dat)):
    template = templates_dat.iloc[[i]].reset_index()
    for j in range(n_gen):
        prompt = template['template_text'][0]

        # randomly select values relevant to all categories
        likely_choice = random.choice(LIKELY)
        comparative_choice = random.choice(COMPARATIVE)
        prob_choice = random.choice(PROB)
        n_med_choice = random.choice(N_MED)
        n_sm_choice = random.choice(N_SM)
        n_hi_choice = random.choice(N_HI)

        ans_idx = None
        if '{{LIKELY}}' in prompt:
            ans_idx = likely_choice[1]
        elif '{{COMPARATIVE}}' in prompt:
            ans_idx = comparative_choice[1]
        else:
            ans_idx = None

        # insert values relevant to just one specific category
        if template['template_type'][0] == 'dice':
            dice_choice = random.choice(DICE)
            prompt = prompt.replace("{{DICE_SIDES}}", dice_choice[0])
            prompt = prompt.replace("{{DICE_VAL}}", p.number_to_words(random.choice(range(1, dice_choice[1]+1))))
        if template['template_type'][0] == 'cards':
            card_num_choice = random.choice(CARD_NUM)
            card_suit_choice = random.choice(CARD_SUIT)
            prompt = prompt.replace("{{CARD_NUM}}", card_num_choice)
            prompt = prompt.replace("{{CARD_SUIT}}", card_suit_choice)
        if template['template_type'][0] == 'sack_items':
            sack_item_choice = random.choice(SACK_ITEM)
            c1 = random.choice(COLORS)
            c2 = random.choice([x for x in COLORS if x != c1])
            prompt = prompt.replace("{{SACK_ITEM}}", sack_item_choice)
            prompt = prompt.replace("{{COLOR_1}}", c1)
            prompt = prompt.replace("{{COLOR_2}}", c2)
        if template['template_type'][0] == 'coin':
            coin_choice = random.choice(COIN)
            other_coin_choice = "heads" if coin_choice == "tails" else "tails"
            prompt = prompt.replace("{{COIN}}", coin_choice)
            prompt = prompt.replace("{{OTHER_COIN}}", other_coin_choice)

        # sub in the values
        prompt = prompt.replace("{{LIKELY}}", likely_choice[0])
        prompt = prompt.replace("{{PROB}}", prob_choice)
        prompt = prompt.replace("{{COMPARATIVE}}", comparative_choice[0])
        prompt = prompt.replace("{{N_MED}}", p.number_to_words(n_med_choice))
        prompt = prompt.replace("{{N_SM}}", p.number_to_words(n_sm_choice))
        prompt = prompt.replace("{{N_HI}}", p.number_to_words(n_hi_choice))
        prompt = prompt.replace("{{NAME}}", random.choice(names_list))

        # finally, fix some weird a/an issues that I don't know a better way to fix. yay regex.
        prompt = re.sub(r"(\s[aA])(\s[aAeEiIoOuU])", r"\1n\2", prompt)
        # lol, but keep 'a one'
        prompt = prompt.replace(" an one", " a one")

        # create next row of dataframe
        this_idx = (i * n_gen) + j
        dat['prompt'][this_idx] = prompt
        dat['answer_index'][this_idx] = ans_idx

dat.drop_duplicates().to_csv("gamblers_fallacy_templated.csv")

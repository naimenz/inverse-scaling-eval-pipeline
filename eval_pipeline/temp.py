import pandas as pd
import json
with open("/home/ian/code/lm_internship/eval-pipeline/698gpt3.cache") as f:
    original = json.load(f)
with open("/home/ian/code/lm_internship/eval-pipeline/gpt3.cache") as f:
    later = json.load(f)

print(len(original) +  len(later))
all_losses = {**original, **later}
rows = [{"text": text, **losses} for text, losses in all_losses.items()]
df = pd.DataFrame.from_records(rows)
print(len(df))
df.to_csv("/home/ian/code/lm_internship/eval-pipeline/data/conjunction_fallacy/losses_gpt3.csv")

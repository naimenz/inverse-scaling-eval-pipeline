#!/bin/bash

source .venv/bin/activate
python eval_pipeline/main.py \
--read_path data/extension_fallacy/filled_templates.csv \
--write_path data/extension_fallacy/losses_gpt2.csv \
--sizes gpt2 gpt2-medium gpt2-large

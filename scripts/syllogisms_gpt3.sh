#!/bin/bash

source .venv/bin/activate
python eval_pipeline/main.py \
--read-path data/syllogism/filled_templates_sample.csv \
--write-path data/syllogism/losses_gpt3_sample.csv \
--sizes ada babbage curie
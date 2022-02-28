#!/bin/bash

source .venv/bin/activate
python eval_pipeline/plot_loss.py \
--read-path data/extension_fallacy/losses_gpt2.csv \
--write-path data/extension_fallacy/fig_losses_gpt2.png

source .venv/bin/activate

EXP_DIR=shell-conj-test
python eval_pipeline/main.py \
    --dataset conjunction_fallacy \
    --exp-dir $EXP_DIR \
    --models gpt2 gpt2-medium \
    --use-gpu \
&& \
python eval_pipeline/plot_loss.py \
    $EXP_DIR



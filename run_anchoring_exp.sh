source .venv/bin/activate

EXP_DIR=shell-anchoring-test
python eval_pipeline/main.py \
    --dataset anchoring \
    --task-type numeric \
    --exp-dir $EXP_DIR \
    --models ada babbage curie davinci \
    --batch-size 9 \
&& \
python eval_pipeline/evaluate_anchoring.py \
    $EXP_DIR \
&&
python eval_pipeline/plot_loss.py \
    --task-type numeric \
    $EXP_DIR



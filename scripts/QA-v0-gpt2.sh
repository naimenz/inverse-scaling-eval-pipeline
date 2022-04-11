SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=QA-v0-gpt2
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset QA_bias-v0 \
    --task-type QA \
    --exp-dir $EXP_DIR \
    --models gpt2 gpt2-medium\
    --use-gpu \
&& \
python $SCRIPT_DIR/../eval_pipeline/evaluate_QA.py \
    $EXP_DIR \
&&
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type numeric \
    $EXP_DIR



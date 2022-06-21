SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=QA-v2-gpt2-new
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset QA_bias-v2 \
    --task-type logodds \
    --exp-dir $EXP_DIR \
    --models gpt2 gpt2-medium \
    --logging-level debug \
    --use-gpu \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type logodds \
    $EXP_DIR



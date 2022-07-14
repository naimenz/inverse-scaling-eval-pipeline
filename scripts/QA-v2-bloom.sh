SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=QA-v2-bloom
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset QA_bias-v2 \
    --task-type logodds \
    --exp-dir $EXP_DIR \
    --models bloom-350m bloom-760m \
    --logging-level info \
    --use-gpu \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type logodds \
    $EXP_DIR



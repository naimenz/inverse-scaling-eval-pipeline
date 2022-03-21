SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=shell-anchoring-test
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset anchoring \
    --task-type numeric \
    --exp-dir $EXP_DIR \
    --models ada babbage \
    --batch-size 9 \
&& \
python $SCRIPT_DIR/../eval_pipeline/evaluate_anchoring.py \
    $EXP_DIR \
&&
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type numeric \
    $EXP_DIR



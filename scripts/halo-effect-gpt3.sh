SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=halo-effect-gpt3
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset halo-effect \
    --task-type QA \
    --exp-dir $EXP_DIR \
    --models ada babbage curie davinci \
    --batch-size 100 \
&& \
python $SCRIPT_DIR/../eval_pipeline/evaluate_QA.py \
    $EXP_DIR \
&&
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type numeric \
    $EXP_DIR



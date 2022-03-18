SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=lambada_gpt3
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset lambada \
    --exp-dir $EXP_DIR \
    --models curie davinci \
    --batch-size 50 \
    --task-type lambada \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    $EXP_DIR



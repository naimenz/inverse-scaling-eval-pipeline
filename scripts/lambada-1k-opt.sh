SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=lambada-1k_opt
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset lambada-1k \
    --exp-dir $EXP_DIR \
    --models opt-125m opt-350m \
    --task-type single_word \
    --use-gpu \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type single_word \
    $EXP_DIR



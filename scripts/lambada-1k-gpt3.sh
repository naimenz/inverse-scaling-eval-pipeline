SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=lambada-1k_gpt3
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset lambada-1k \
    --exp-dir $EXP_DIR \
    --models ada babbage curie davinci \
    --batch-size 50 \
    --task-type single_word \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type single_word \
    $EXP_DIR



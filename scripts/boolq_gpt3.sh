SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=boolq-gpt3
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset boolq \
    --exp-dir $EXP_DIR \
    --models ada babbage curie davinci \
    --batch-size 150 \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    $EXP_DIR



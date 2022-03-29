SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=dice-0shot-gpt3
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset dice-0shot \
    --exp-dir $EXP_DIR \
    --models ada babbage curie davinci \
    --batch-size 200 \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type classification_acc \
    $EXP_DIR



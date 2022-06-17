SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=syllogism-0shot-gpt2-seq
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset syllogism \
    --exp-dir $EXP_DIR \
    --models gpt2 gpt2-medium \
    --use-gpu \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    $EXP_DIR



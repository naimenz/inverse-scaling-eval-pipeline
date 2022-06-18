SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=syllogism-0shot-opt
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset syllogism \
    --exp-dir $EXP_DIR \
    --models opt-125m opt-350m opt-1.3b opt-2.7b \
    --use-gpu \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    $EXP_DIR



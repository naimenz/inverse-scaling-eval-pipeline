SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=syllogism-opt
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset syllogism \
    --exp-dir $EXP_DIR \
    --models opt-125m opt-350m \
    --use-gpu \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    $EXP_DIR \
  --task-type classification_loss


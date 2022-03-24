SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=anli_all-acc-gpt2
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset anli_all \
    --exp-dir $EXP_DIR \
    --task-type classification \
    --models gpt2 gpt2-medium \
    --use-gpu \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    $EXP_DIR



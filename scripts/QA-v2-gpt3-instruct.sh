SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=QA-v2-gpt3
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset QA_bias-v2 \
    --task-type logodds \
    --exp-dir $EXP_DIR \
    --models text-ada-001 text-babbage-001 text-curie-001 text-davinci-001 \
    --batch-size 20 \
&&
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type logodds \
    $EXP_DIR



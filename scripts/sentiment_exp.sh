SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=sentiment-analysis-gpt2
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset sentiment_analysis \
    --exp-dir $EXP_DIR \
    --models gpt2 gpt2-medium gpt2-large \
    --use-gpu \
&& \
python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    $EXP_DIR



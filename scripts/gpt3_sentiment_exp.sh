SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

EXP_DIR=sentiment-analysis-gpt3
python $SCRIPT_DIR/../eval_pipeline/main.py \
    --dataset sentiment_analysis \
    --exp-dir $EXP_DIR \
    --models ada babbage curie davinci \
    --batch-size 50 \
&& \
python eval_pipeline/plot_loss.py \
    $EXP_DIR



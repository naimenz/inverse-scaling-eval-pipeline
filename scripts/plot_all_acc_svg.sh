SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

for dir in $SCRIPT_DIR/../results/*/
do
    dir=${dir%*/}
    exp_dir="${dir##*/}"
    echo $exp_dir
    python $SCRIPT_DIR/../eval_pipeline/plot_loss.py \
    --task-type classification_acc \
    --no-show \
    $exp_dir
done

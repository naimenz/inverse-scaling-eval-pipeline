SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../.venv/bin/activate

for dir in $SCRIPT_DIR/../results/*/
do
    dir=${dir%*/}
    exp_dir="${dir##*/}"
    cp $dir/loss_plot.svg $SCRIPT_DIR/../docs/assets/images/$exp_dir.svg
done

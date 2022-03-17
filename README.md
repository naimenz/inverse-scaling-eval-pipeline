# About
This repo is for running inverse scaling examples.
There is a colab set up for it, which you can find in the task spreadsheet.

# Running on NYU
To run on NYU:
## Installing
1. Follow [these Getting Started instructions](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/getting-started?authuser=0) to get connected to Greene.
2. Follow [these Singularity instructions](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda?authuser=0) up until **Install packages** with the following differences:
    1. Instead of `cuda11.2-cudnn8-devel-ubuntu20.04.sif`, use `cuda11.3.0-cudnn8-devel-ubuntu20.04.sif`
    2. Instead of `overlay-7.5GB-300K.ext3.gz` use `overlay-10GB-400K.ext3`
3. Activate the Singularity image with the overlay
    1. Remember to run `source /ext3/env.sh` (or whatever you called it when setting up the image) to activate the Python environment.
4. `cd` to `/ext3` and run `git clone https://github.com/naimenz/inverse-scaling-eval-pipeline` to get a copy of the code.
5. Run `pip install .` to install the `inverse-scaling-eval-pipeline` package.
6. Run the command `python -m pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html` to install the correct version of PyTorch.
## Running
1. Copy the `example.sbatch` script included under the `/ext3/inverse-scaling-eval-pipeline/scripts` directory to somewhere outside the image, e.g. your `home` or `scratch`.
2. There are two options for pointing to your data:
    1. Put your data in `/ext3/inverse-scaling-eval-pipeline/data` and use the option `--data` as in the script.
    2. Put your data elsewhere and use the option `--dataset-path` to point to it.
3. For `--exp-dir`, give the *absolute* path of the directory you want the results to be saved in.
4. Remember to add the flag `--use-gpu` only for HuggingFace models (GPT-2, GPT-Neo) and to add the flag `--batch-size n` (with n > 1) only for OpenAI API models (GPT-3)
5. Submit your `.sbatch` file as a job with `sbatch example.sbatch`
6. Run the plotting file by activating the Singularity image and running `python /ext3/inverse-scaling-eval-pipeline/eval_pipeline/plot_loss.py </path/to/results/dir>`
---
Let me know which parts of these instructions are incorrect/unclear!


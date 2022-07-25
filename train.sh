#!/bin/bash

#$-o ./log/train.log
#$-l rt_G.small=1
#$-l h_rt=24:00:00
#$-j y
#$-cwd

## >>> conda init >>>

__conda_setup="$(CONDA_REPORT_ERRORS=false '$HOME/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"

if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate ml
    else
        \export PATH="$PATH:$HOME/anaconda3/bin"
    fi
fi
unset __conda_setup
## <<< conda init <<< 

## Activation
conda activate ml
source /etc/profile.d/modules.sh
module load cuda/11.2/11.2.2 cudnn/8.2/8.2.1 nccl/2.8/2.8.4-1
conda activate ml
cd /home/acd13642rm/project_me/CLIP
python main.py -w --train --inference --save_weight
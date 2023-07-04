#!/bin/bash

# source /home/virtual_envs/ml/bin/activate
# pipenv shell

# nvidia-smi

echo "You would require an entire dataset to train using this script."

python3 main.py -latent=512 -experiment_run='0112_topkk=9_fa=0.001_random=4' \
                        -base="vgg16" \
                        -last_layer_weight=-1 \
                        -fa_coeff=0.001 \
                        -topk_k=9 \
                        -train_dir="/Users/ruchit/urop/data/train/" \
                        -push_dir="/Users/ruchit/urop/data/prototypes/" \
                        -test_dir="/Users/ruchit/urop/data/validation/" \
                        -random_seed=4 \
                        # -finer_dir="/usr/xtmp/mammo/Lo1136i_finer/by_margin/train_augmented_250/" \
                        # -model="/Users/ruchit/urop/model_0/9nopush1.0000.pth"

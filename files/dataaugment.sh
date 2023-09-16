#!/bin/bash

# source /home/virtual_envs/ml/bin/activate
pipenv shell

srun -u python dataHandling.py

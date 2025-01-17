#!/bin/sh

# Set CUDA_HOME to the path where CUDA is installed
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=6,7

echo "The script has started"

# Run your Python script
# python finetune_RoLlama2.py
# python evaluation.py
python evaluation_base.py
# python inference_RoLlama2_finetuned.py


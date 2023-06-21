#!/bin/bash

echo 'Date: ' `date` 
echo 'Host: ' `hostname` 
echo 'System: ' `uname -spo` 
echo 'GPU: ' `nvidia-smi --query-gpu=gpu_name --format=csv,noheader`
set -e

export HOME=$PWD
cp -r /staging/groups/cafa-5-group/Training-Looped-Transformers $HOME/Training-Looped-Transformers
cd $HOME/Training-Looped-Transformers

# Prepare environments
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Set up conda
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no

# Install packages specified in the environment file
conda env create -f environment.yml --quiet

# Activate the environment and log all packages that were installed
conda activate loop-trans

cd scripts
bash train.sh

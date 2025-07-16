#!/bin/bash
YOUR_PATH_TO_CONDA=~/anaconda3

# MuJoCo Setup
sudo apt update
sudo apt install g++
sudo apt-get upgrade libstdc++6
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libghc-x11-dev
sudo apt install libcairo2-dev pkg-config python3-dev
sudo apt-get install patchelf

wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210_linux.tar.gz
mkdir ~/.mujoco
tar -zxvf mujoco210_linux.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# Conda Environment
conda env create -f environment.yml
conda activate off-moo
conda install gxx_linux-64 gcc_linux-64
conda install --channel=conda-forge libxcrypt

# Install dependencies
pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Additional dependencies
pip install scipy==1.10.1
pip install scikit-learn==0.21.3
pip install --upgrade pandas
pip install --upgrade kiwisolver

# Diffusion dependencies
pip install gin-config==0.5.0
pip install einops==0.8.0
pip install torchdiffeq==0.2.5
pip install pygmo==2.19.5
pip install accelerate==1.0.1
pip install wandb==0.19.6

# Sklearn==0.21.3 might be wrong in some scenarios, thus we fix bugs with the scripts below.
# Please set your path to conda here
bash fix_contents.sh ${YOUR_PATH_TO_CONDA}/envs/off-moo/lib/python3.8/site-packages/sklearn/cross_decomposition/pls_.py "pinv2" "pinv"

# MuJoCo Setup
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
conda activate off-moo

# For MuJoCo, if it raises an error that .h file is not found, an easy way is to copy that from /usr/include
mkdir ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/X11
mkdir ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/GL
sudo cp /usr/include/X11/*.h ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/X11/
sudo cp /usr/include/GL/*.h ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/GL

# Sci-Design Setup
pip show hydra-core==1.3.2
pip install omegaconf==2.3.0

cd off_moo_bench/problem/lambo
pip install -e .
# This takes some time approx. 30 minutes
python scripts/black_box_opt.py optimizer=mf_genetic optimizer/algorithm=nsga2 task=proxy_rfp tokenizer=protein 
cd ../../../

# Check if everything is correct
python tests/test_mujoco.py
python tests/test_env.py

#  No NAS stuff for now
# python config_evoxbench.py 
# python tests/test_vallina_modules.py

# Make sure that you have paste your Google Drive API below
# curl -H "Authorization: Bearer <Your Google Drive APIs>" https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view?usp=sharing -o database.zip
# curl -H "Authorization: Bearer <Your Google Drive APIs>" https://drive.google.com/file/d/1r0iSCq1gLFs5xnmp1MDiqcqxNcY5q6Hp/view?usp=sharing -o data.zip

# unzip database.zip -d off_moo_bench/problems/mo_nas/
# unzip data.zip -d off_moo_bench/problems/mo_nas/

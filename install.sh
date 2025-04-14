#!/bin/bash

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p $HOME/miniconda3

# Initialize conda
conda init
conda create --name p3.9 python=3.9
conda activate p3.9

# Install Python dependencies
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Clone and set up Kaolin
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu124


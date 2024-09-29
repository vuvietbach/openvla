#!/bin/bash

# Download Miniconda installer
echo "Downloading Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh

DIR=$HOME
# Run the installer
echo "Running the Miniconda installer..."
bash Miniconda3.sh -b -p $DIR/miniconda3

# Initialize Miniconda
echo "Initializing Miniconda..."
eval "$($DIR/miniconda3/bin/conda shell.bash hook)"
conda init

# Clean up
echo "Cleaning up..."
rm Miniconda3.sh

# Finish
echo "Miniconda installation completed!"

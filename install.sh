#!/usr/bin/env bash
# Author: Morgan Stuart

INSTALL_PATH=$(pwd)

echo "This script will install a virtual environment here:"
echo "$INSTALL_PATH/venv"

read -p "Continue (y/n)?" choice
case "$choice" in
    n|N) echo "Exiting..."; exit 0;
esac

#TODO: check if venv exists?

pushd .

cd $INSTALL_PATH

#sudo apt-get install python3-pip build-essential libssl-dev libffi-dev python3-dev

echo "Creating virtual environment"
python3 -m venv ./venv

echo "Activating new virtual environment"
source ./venv/bin/activate

echo "Installing dependencies"
# Sometimes setup script with 'develop' doesn't install depends?
pip install numpy pandas theano keras sklearn tensorflow

# Pytorch for python 3.6
#pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl

#Pytorch for python 3.5
#pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl

echo "Install the SemEval Task 8 Package"
python ./setup.py develop


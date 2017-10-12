#!/usr/bin/env bash
# Author: Morgan Stuart

INSTALL_PATH=$(pwd)

echo "This script will install a virtual environment here:"
echo "$INSTALL_PATH/venv"

read -p "Continue (y/n)?" choice
case "$choice" in
    n|N) echo "Exiting..."; exit 0;
esac

pushd .

cd $INSTALL_PATH

echo "Continue!"
#sudo apt-get install python3-pip build-essential libssl-dev libffi-dev python3-dev

python3 -m venv ./venv

source ./venv/bin/activate

pip install numpy
pip install pandas keras theano scikit-learn


python ./setup.py develop


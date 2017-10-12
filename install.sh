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

#sudo apt-get install python3-pip build-essential libssl-dev libffi-dev python3-dev

echo "Creating virtual environment"
python3 -m venv ./venv

echo "Activating new virtual environment"
source ./venv/bin/activate

echo "Install the SemEval Task 8 Package"
python ./setup.py develop


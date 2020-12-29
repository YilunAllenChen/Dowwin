export PYTHONPATH=$PWD

if [ -z "$1" ]
then
    echo "Must choose a target in the Makefile! Example: './run.sh dae' for running dae module."
else
    make $1
fi

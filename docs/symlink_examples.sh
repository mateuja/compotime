#!/bin/sh

# This script was adapted from https://github.com/sktime/sktime/blob/main/docs/symlink_examples.sh

rm -rf source/_examples

mkdir source/_examples

cd source/_examples/ || return

ln -s ../../../examples/*.ipynb .

cd - || exit

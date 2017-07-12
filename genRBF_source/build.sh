#!/bin/bash

python setup.py build_ext --inplace

rm -r ./build/
rm cRBFkernel.cpp

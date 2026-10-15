#! /bin/bash

if [[ $1 =~ ^[0-9]+$ ]]; then
	nvcc tileMatrixMult.cu -DTILE_WIDTH=$1  -o tileMatrixMult
else
	echo "Please enter tile width as first argument!"
fi

#!/bin/bash

python procam.py syn_proj \
	-t "./data/single_pixel.png" \
	-l "./data/matrix.hdf5" \
	-m "./models/VGG19_normalized_avg_pool_pytorch" \
	-o "./output" \
	--batch 3 \
	--layer 0 1 \
	--bright 5.0 --full_out False --step 10

# increase --step to 500 if you have a powerful machine (GPU) and want to see a converged image!

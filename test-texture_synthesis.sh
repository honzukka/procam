#!/bin/bash

python procam.py syn_tex \
	-t "./data/big_pebbles.png" \
	-b "./data/carpet.png" \
	-m "./models/VGG19_normalized_avg_pool_pytorch" \
	-o "./output" \
	--pyramid_w 1.0 1.0 1.0 \
	--bright 5.0 --full_out False --step 10

# increase --step to 500 if you have a powerful machine (GPU) and want to see a converged image!

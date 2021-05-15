#!/bin/bash

python procam.py comp_tex \
	-t "./data/pebbles.jpg" \
	-b "./data/photo.jpg" \
	-o "./output" \
	--bright 5.0 --full_out False --step 50

python procam.py syn_tex \
    -t "./data/pebbles.jpg" \
    -b "./data/photo.jpg" \
    -m "./models/VGG19_normalized_avg_pool_pytorch" \
    -o "./output" \
    --pyramid_w 1.0 1.0 1.0 \
	--bright 5.0 --full_out False --step 100

#!/bin/bash

python procam.py comp_tex \
	-t "./data/big_pebbles.png" \
	-b "./data/carpet.png" \
	-o "./output" \
	--bright 1.0 --full_out False --step 50

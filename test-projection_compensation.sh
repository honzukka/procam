#!/bin/bash

python procam.py comp_proj \
	-t "./data/color_palette.png" \
	-l "./data/matrix.hdf5" \
	-o "./output" \
	--bright 5.0 --full_out False --step 50

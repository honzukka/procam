#!/bin/bash

# Handles everything from rendering basis images in Mitsuba
# to building a matrix out of them.

# job parameters
# ---------------------------------------------------
result_dir="${PWD}/data"    # directory for input/output files
tmp_dir="${PWD}/tmp"        # temporary directory for computations
script_dir="${PWD}"         # directory with necessary Python source files
size_x=5                    # horizontal projector resolution
size_y=3                    # vertical projector resolution
# ---------------------------------------------------

echo "Initializing working directory: ${tmp_dir}..."
mkdir "$tmp_dir"

echo "Copying input files to ${tmp_dir}..."
cp -r "${result_dir}"/*/ "${tmp_dir}"	                # assets (=subdirs in data dir, if any)
cp "${result_dir}/scene.xml" "${tmp_dir}/"              # scene file

echo "Rendering bases..."
rendering_start="$SECONDS"
${script_dir}/render_bases.sh \
                        -d "$tmp_dir" \
                        -p "$script_dir" \
                        -i "$size_x" -j "$size_y" \
                        -a "$size_x" -b "$size_y" && \
    let "rendering_delta = $SECONDS - $rendering_start" && \
    echo -e "Rendering done in ${rendering_delta}s.\n" || \
    exit 1

echo "Collecting basis images into a matrix..."
building_start="$SECONDS"
python "${script_dir}/build_matrix.py" "$tmp_dir" && \
    let "building_delta = $SECONDS - $building_start" && \
    echo -e "Matrix built in ${building_delta}s.\n" ||
    exit 1

echo "Saving result to ${result_dir} and cleaning up..."
mv "${tmp_dir}/matrix.hdf5" "${result_dir}/"
rm -r "${tmp_dir}/"*

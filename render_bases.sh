#!/bin/bash

if [[ $OSTYPE != "linux-gnu"* ]] && [[ $OSTYPE != "darwin"* ]]; then
    echo >&2 "This script isn't ready for your operating system!"
    exit 1
fi

# configure the job, i.e. specify which basis textures will be rendered and where
# the basic idea is to work in rectangular blocks,
# so that many jobs can be run simultaneously on different nodes
# ------------------------------------------------------------------------------
tex_res_x=3
tex_res_y=3
pos_start_x=1
pos_start_y=1
range_x=$tex_res_x
range_y=$tex_res_y
data_dir=""
py_script_dir=""

data_dir_docker="/mitsuba/data"     # this is what data_dir is mounted at inside Docker

while getopts 'i:j:x:y:a:b:d:p:' c
do
    case $c in
        i) tex_res_x=$OPTARG ;;
        j) tex_res_y=$OPTARG ;;
        x) pos_start_x=$OPTARG ;;
        y) pos_start_y=$OPTARG ;;
        a) range_x=$OPTARG ;;
        b) range_y=$OPTARG ;;
	    d) data_dir=$OPTARG ;;
        p) py_script_dir=$OPTARG ;;
    esac
done

if [ -z "$data_dir" ]; then
    echo >&2 "Data directory not specified!"
    exit 1
fi

if [ -z "$py_script_dir" ]; then
    echo >&2 "Python script directory not specified!"
    exit 1
fi

mts_cmd="mitsuba"
docker_run="docker run --rm -v "${data_dir}:${data_dir_docker}" mitsuba"

mts_script="${data_dir}/mts_script${pos_start_x}-${pos_start_y}.sh"

let "pos_end_x = $pos_start_x + $range_x - 1"
let "pos_end_y = $pos_start_y + $range_y - 1"

# make sure POS_END doesn't overflow
if [ $pos_end_x -gt $tex_res_x ]; then
    pos_end_x=$tex_res_x
fi

if [ $pos_end_y -gt $tex_res_y ]; then
    pos_end_y=$tex_res_y
fi
# ------------------------------------------------------------------------------

echo "Preparing bases from [${pos_start_x}, ${pos_start_y}] to [${pos_end_x}, ${pos_end_y}]..."
: > "${data_dir}/scene_files.out"
for x in `seq $pos_start_x $pos_end_x`
do
    for y in `seq $pos_start_y $pos_end_y`
    do
        # make a separate directory for each rendering
        # (allows us to recycle scene files,
        # otherwise we would have to change the basis texture name in each of them)
        printf -v x_padded "%04d" "$x"
        printf -v y_padded "%04d" "$y"
        new_dir="${data_dir}/${x_padded}-${y_padded}"
        new_dir_docker="${data_dir_docker}/${x_padded}-${y_padded}"
        if [ ! -d "$new_dir" ]; then
            mkdir "$new_dir"
        fi

        # generate basis texture
        python "${py_script_dir}/basis_texture.py" \
                                        --res $tex_res_x $tex_res_y \
                                        --pos $x $y \
                                        "${new_dir}/basis_texture.png" || exit 1
        # copy scene file & update relative paths to assets
        # TODO: make the relative path update more robust
        cp "${data_dir}/scene.xml" "${new_dir}/scene.xml"
        sed -r -i '' -e 's/([a-z]+\/)/..\/\1/g' "${new_dir}/scene.xml"

        # accumulate arguments for Mitsuba
        echo "${new_dir_docker}/scene.xml" >> "${data_dir}/scene_files.out"
    done
done

# use xargs when running Mitsuba in case the number of scene files is way too large
echo "cat \"${data_dir_docker}/scene_files.out\" | xargs \"$mts_cmd\" -q" > "$mts_script"

echo "Running Mitsuba..."
$docker_run bash "${data_dir_docker}/mts_script1-1.sh"

# clean up
rm "$mts_script"

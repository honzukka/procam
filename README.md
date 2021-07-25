# Procam

Procam is a Python program implementing a novel technique for [projection mapping](https://en.wikipedia.org/wiki/Projection_mapping). It accompanies my Master's thesis which explains the technique in depth and includes experiments and results. The thesis is located here: [https://github.com/honzukka/Masters-Thesis](https://github.com/honzukka/Masters-Thesis).

## What does Procam do?

First of all, it does **projection mapping**. To do projection mapping, we need a scene:

![intro - scene](/readme_images/intro-scene.jpg)

Then we would like to use a projector to alter the appearance of the scene. This is what we would like the scene to look like:

![intro - desired appearance](/readme_images/intro-desired_appearance.jpg)

But if we take the image and project it directly, the resulting appearance will not be what we expect:

![intro - actual appearance](/readme_images/intro-actual_appearance.jpg)

This is caused by the scene having non-uniform reflectance properties (unlike a cinema screen). Notice also, that projectors have limited brightness which limits us further (the projection image above is not as saturated as the original). Projection mapping is about computing a compensation image which results in the desired appearance when projected onto our scene:

![intro - compensated appearance](/readme_images/intro-compensated_appearance.jpg)

Projection mapping has been around for a long time. It is usually done with the help of projector-camera systems (hence Procam). Camera images of scene appearance are used to alter the brightness of each pixel of the projected image so that the appearance of that pixel, once projected, is as close to what we want it to be as possible:

![intro - projection mapping](/readme_images/intro-projection_mapping.jpg)

But what if our projector is not bright (or dark) enough to push a particular pixel to where it needs to be?

Our method attempts to account for this case by modifying the original projection image in such a way that its pixels are easier to compensate, but so that the image still looks like the original one.

In particular, we focus on projecting textures and employ the techniques of [texture synthesis](https://en.wikipedia.org/wiki/Texture_synthesis). For more details, please see the full thesis: [https://github.com/honzukka/Masters-Thesis](https://github.com/honzukka/Masters-Thesis).

## Okay! What do your results look like?

Let us consider the following scene:

![result - scene](/readme_images/result-background.jpg)

It is in fact just a flat wallpaper-like background. This is a simplification that allows us to use faster computations, but it still proves the point. And on top of that, Procam also supports projecting onto arbitrary scenes.

Next, here is the texture that we want to project onto the scene - we want the scene to look like the texture:

![result - texture](/readme_images/result-texture.jpg)

If we use a **conventional** projection mapping method, we obtain the following compensation image (left) and final apperance (right):

![result - baseline compensation](/readme_images/result-baseline_compensation.jpg) ![result - baseline projection](/readme_images/result-baseline_projection.jpg)

Here are results obtained using **our** method:

![result - ours compensation](/readme_images/result-ours_compensation.jpg) ![result - ours projection](/readme_images/result-ours_projection.jpg)

## How can I run it?

1. Clone this repository and switch to its root folder
2. Create a Conda environment: `conda env create -f environment_mac.yml` (only Mac CPU-only version is available, but others should be easy to create, for example by downloading the GPU version of PyTorch)
3. _(optional)_ Run `docker build -t mitsuba .` in order to be able to project onto custom scenes
4. _(optional)_ Run `pytest` to ensure that the environment is properly set up
5. Run `./example.sh` to execute Procam with example arguments and produce the results shown [above](#okay-what-do-your-results-look-like) including the baseline
6. Run `python procam.py -h` for an explanation of what various arguments mean

## Anything else I should know?

Procam has four modes:

* `comp_tex`
* `comp_proj`
* `syn_tex`
* `syn_proj`

First, Procam can use both a baseline conventional projection mapping method (`comp_*`) and our method (`syn_`). Next, it supports a lightweight projection onto a flat background (`*_tex`) as well as a projection onto an arbitrary 3D scene (`*_proj`).

To get a better idea of what each of these modes does, there are four scripts in the root folder in the format of `test-*.sh` which can do a test run of each of the modes.

Last but not least, while supplying an image to project onto in on of the `*_tex` modes is straightforward, how does one supply a custom 3D scene?

An example workflow is provided. After running step 3 from from the [setup guide](#how-can-i-run-it), it is possible to run `./measure_lt.sh`. This script generates an `matrix.hdf5` file from a `scene.xml` file. `matrix.hdf5` is used by Procam to project onto a scene described in `scene.xml` which is a standard Mitsuba scene configuration file.

Custom `scene.xml` files can be created with the help of the Mitsuba Docker container which is provided with Procam. The important part is to place a projector into the scene. This is done using the `projector` emitter type (this is a custom emitter type implemented in [https://github.com/honzukka/mitsuba](https://github.com/honzukka/mitsuba)). This emitter type needs an image which it can project. Before running `measure_lt.sh`, this image needs to be set to `"./basis_texture.png"`, so that `matrix.hdf5` can be properly generated.

More detailed information can be found in the thesis at [https://github.com/honzukka/Masters-Thesis](https://github.com/honzukka/Masters-Thesis).

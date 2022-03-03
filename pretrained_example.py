# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os

import pickle
import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib
import config

def main():

    # Initialize TensorFlow.
    tflib.init_tf()
    # Print network details.
    file = open("karras2019stylegan-ffhq-1024x1024.pkl", 'rb')
    _G, _D, Gs = pickle.load(file)
    Gs.print_layers()

    '''
    # Pick latent vector.
    rnd = np.random.RandomState(5)
    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    i = 0
    for img in images:
        png_filename = os.path.join(config.result_dir, 'example' + str(i) + '.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        i += 1
        
    '''
    rnd = np.random.RandomState(5)
    latent = rnd.randn(1, Gs.input_shape[1])
    print(latent.shape)


    i = 0
    latents = np.random.RandomState(5).randn(sum(3 * 2 ** lod for lod in [0, 1, 2, 2, 3, 3]), Gs.input_shape[1])
    print(latents.shape)
    for latent in latents[0]:
        # Generate image.
        print(latent.shape)
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latent, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'ALI-example' + str(i) + '.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        i += 1



if __name__ == "__main__":
    main()

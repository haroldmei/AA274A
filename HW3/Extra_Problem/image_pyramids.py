#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


def half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    m, n, c = image.shape[0], image.shape[1], image.shape[2]
    down_scaleim = np.zeros((m/2, n/2, c))

    for i in range(m/2):
        for j in range(n/2):
            for k in range(c):
                down_scaleim[i, j, k] = image[2*i, 2*j, k]
    return down_scaleim

    ########## Code ends here ##########


def blur_half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    blur_im = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0.7)

    downscaled_image = half_downscale(blur_im)

    return downscaled_image

    ########## Code ends here ##########


def two_upscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        upscaled_image: A 2x-upscaled version of image.
    """
    ########## Code starts here ##########
    m, n, c = image.shape[0], image.shape[1], image.shape[2]
    double_width = np.zeros((m, 2*n, c))

    for i in range(n):
        for k in range(c):
            double_width[:, 2 * i, k] = image[:, i, k]
            double_width[:, 2 * i + 1, k] = image[:, i, k]

    upscaled_image = np.zeros((2 * m, 2 * n, c))
    for j in range(m):
        for t in range(c):
            upscaled_image[2 * j, :, t] = double_width[j, :, t]
            upscaled_image[2 * j + 1, :, t] = double_width[j, :, t]

    return upscaled_image
    ########## Code ends here ##########


def bilinterp_upscale(image, scale):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
        scale: How much larger to make the image

    Returns
        upscaled_image: A scale-times upscaled version of image.
    """
    m, n, c = image.shape

    f = (1./scale) * np.convolve(np.ones((scale, )), np.ones((scale, )))
    f = np.expand_dims(f, axis=0) # Making it (1, (2*scale)-1)-shaped
    filt = f.T * f

    ########## Code starts here ##########
    print filt

    upwidth_im = np.zeros((m, scale * n, c))

    upscaled_image = np.zeros((scale * m, scale * n, c))

    for i in range(n):
        for k in range(c):
            upwidth_im[:, scale * i, k] = image[:, i, k]

    for j in range(m):
        for t in range(c):
            upscaled_image[scale * j, :, t] = upwidth_im[j, :, t]
    return cv2.filter2D(upscaled_image, -1, filt)
    ########## Code ends here ##########


def main():
    # OpenCV actually uses a BGR color channel layout,
    # Matplotlib uses an RGB color channel layout, so we're flipping the 
    # channels here so that plotting matches what we expect for colors.
    test_card = cv2.imread('test_card.png')[..., ::-1].astype(float)
    favicon = cv2.imread('favicon-16x16.png')[..., ::-1].astype(float)
    test_card /= test_card.max()
    favicon /= favicon.max()

    # Note that if you call matplotlib's imshow function to visualize images,
    # be sure to pass in interpolation='none' so that the image you see
    # matches exactly what's in the data array you pass in.
    
    ########## Code starts here ##########
    down_scale_image = half_downscale(test_card)
    blurdown_scale_image = blur_half_downscale(test_card)
    upscale_im = two_upscale(favicon)
    bilin_upscale_im = bilinterp_upscale(favicon, 8)

    for i in range(2):
        down_scale_image = half_downscale(down_scale_image)
        blurdown_scale_image = blur_half_downscale(blurdown_scale_image)
        upscale_im = two_upscale(upscale_im)


    plt.imshow(down_scale_image)
    plt.show()
    plt.imshow(blurdown_scale_image)
    plt.show()
    plt.imshow(upscale_im)
    plt.show()
    plt.imshow(bilin_upscale_im)
    plt.show()

    ########## Code ends here ##########


if __name__ == '__main__':
    main()

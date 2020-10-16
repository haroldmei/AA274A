#!/usr/bin/env python

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########
    m, n, c  = I.shape[0], I.shape[1], I.shape[2]
    G = np.zeros((m, n))
    k, l = F.shape[0], F.shape[1]
    pad_height = m + 2*(k/2)
    pad_width = n + 2*(l/2)
    padim = np.zeros((pad_height, pad_width, c))

    f = np.reshape(F, (1, k*l*c))

    # zero padding image
    for i in range(m):
       for j in range(n):
           for t in range(c):
               padim[(i+(k/2)), (j+(k/2)), t] = I[i, j, t]

    for i in range(m):
        for j in range(n):
            a = padim[i:i+2*(k/2)+1]
            if l%2 == 0:
                tij = a[:, j:j + 2 * (l / 2)]
            else:
                tij = a[:, j:j+2*(l/2)+1]

            tij = np.reshape(tij, (k*l*c, 1))

            G[i, j] = np.dot(np.array(f), np.array(tij))
    return G


    ########## Code ends here ##########


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########
    m, n, c  = I.shape[0], I.shape[1], I.shape[2]
    G = np.zeros((m, n))
    k, l = F.shape[0], F.shape[1]
    pad_height = m + 2*(k/2)
    pad_width = n + 2*(l/2)
    padim = np.zeros((pad_height, pad_width, c))

    f = np.reshape(F, (1, k*l*c))

    for i in range(m):
       for j in range(n):
           for t in range(c):
               padim[(i+(k/2)), (j+(k/2)), t] = I[i, j, t]

    for i in range(m):
        for j in range(n):
            a = padim[i:i+2*(k/2)+1]
            if l%2 == 0:
                tij = a[:, j:j + 2 * (l / 2)]
            else:
                tij = a[:, j:j+2*(l/2)+1]

            tij = np.reshape(tij, (k*l*c, 1))
            if(np.linalg.norm(tij) == 0):
                G[i, j] = 0
            else:
                G[i, j] = np.dot(np.array(f), np.array(tij))/(np.linalg.norm(f) * np.linalg.norm(tij))
    return G

    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 200, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        stop = time.time()
        print 'Correlation function runtime:', stop - start, 's'
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


def template_match(template, image,
                   num_upscales=2, num_downscales=3,
                   detection_threshold=0.93):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        num_upscales: How many times to 2x-upscale image with Gaussian blur before template matching over it.
        num_downscales: How many times to 0.5x-downscale image with Gaussian blur before template matching over it.
        detection_threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    result = []
    hight, width, _ = template.shape
    res = cv2.matchTemplate(image, template, method=cv2.TM_CCORR_NORMED)
    for q in range(res.shape[0]):
        for w in range(res.shape[1]):
            if res[q][w] >= detection_threshold:
                result.append((q, w, hight, width))

    down_scale = cv2.pyrDown(image)
    for a in range(num_downscales):
        corr_down = cv2.matchTemplate(down_scale,template, method=cv2.TM_CCORR_NORMED)
        for x in range(corr_down.shape[0]):
            for y in range(corr_down.shape[1]):
                if corr_down[x][y] >= detection_threshold:
                    result.append((int(x/(0.5**(a+1))), int(y/(0.5**(a+1))), int(hight/(0.5**(a+1))), int(width/(0.5**(a+1)))))
        down_scale = cv2.pyrDown(down_scale)

    up_scale = cv2.pyrUp(image)
    for b in range(num_upscales):
        corr_up = cv2.matchTemplate(up_scale, template, method=cv2.TM_CCORR_NORMED)
        for i in range(corr_up.shape[0]):
            for j in range(corr_up.shape[1]):
                if corr_up[i][j] >= detection_threshold:
                   result.append((int(i/(2**(b+1))), int(j/(2**(b+1))), int(hight/(2**(b+1))), int(width/(2**(b+1)))))
        up_scale = cv2.pyrUp(up_scale)

    return result
    ########## Code ends here ##########


def create_and_save_detection_image(image, matches, filename="image_detections.png"):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.

    Returns
        None, this function will save the detection image in the current folder.
    """
    det_img = image.copy()
    for (y, x, bbox_h, bbox_w) in matches:
        cv2.rectangle(det_img, (x, y), (x + bbox_w, y + bbox_h), (255, 0, 0), 2)

    cv2.imwrite(filename, det_img)


def main():
    template = cv2.imread('messi_face.jpg')
    image = cv2.imread('messipyr.jpg')

    matches = template_match(template, image)
    create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, detection_threshold=0.87)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)


if __name__ == '__main__':
    main()

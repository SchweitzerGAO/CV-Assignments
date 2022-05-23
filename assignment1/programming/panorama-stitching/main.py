import cv2
import numpy as np
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_name', default=None, type=str, nargs='+')
    parser.add_argument('--show_match', default=False, type=str2bool)
    return parser


# Step 1: detect SIFT keypoints and build descriptors
def detectKeypoints(img):
    """
    detect SIFT keypoints, build descriptors using OpenCV API

    :param img: the image
    :return: the keypoints and descriptors
    """
    # convert to grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create SIFT detector and detect the keypoints and build descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(gray_image, None)

    # convert the keypoints into numpy array
    keypoints = np.array([point.pt for point in keypoints], dtype=np.float32)

    return keypoints, descriptor


# Step 2 & 3: build correspondence using KNN and solve homography matrix
def solve_H(keypoints_1, descriptor_1, keypoints_2, descriptor_2, ratio, threshold):
    """
    build correspondence for keypoints in 2 images and solve the homography matrix

    :param keypoints_1: the keypoints of image 1
    :param descriptor_1: the descriptors of image 1
    :param keypoints_2: the keypoints of image 2
    :param descriptor_2: the descriptors of image 2
    :param ratio: if the ratio of nearest distance and the second-nearest distance(using KNN algorithm here)
                  is less than this, mark the point pair as matched
    :param threshold: the threshold for RANSAC sampling
    :return: corresponding pairs, homography matrix and the homography matching status(True or False)
    """
    # create a brute-force matcher and use KNN algorithm to rawly match points
    KNN_matcher = cv2.BFMatcher()
    raw_correspondence = KNN_matcher.knnMatch(descriptor_1, descriptor_2, k=2)
    filtered_correspondence = []

    # filter the correspondence using the parameter 'ratio'
    for rc in raw_correspondence:
        if len(rc) == 2 and rc[0].distance < rc[1].distance * ratio:  # ensure that there are 2 matching points
            # append the filtered correspondence,where the trainIdx and queryIdx stands for the index of matching points
            # in image 2 and 1
            filtered_correspondence.append((rc[0].trainIdx, rc[0].queryIdx))

    # to solve homography matrix, the number of point pairs shall >= 4
    if len(filtered_correspondence) > 4:
        # the corresponding points in image 1 and 2
        correspond_1 = np.array([keypoints_1[i] for (_, i) in filtered_correspondence], dtype=np.float32)
        correspond_2 = np.array([keypoints_2[i] for (i, _) in filtered_correspondence], dtype=np.float32)
        # find the homography matrix and the success status
        H, success = cv2.findHomography(correspond_1, correspond_2, cv2.RANSAC, threshold)
        return filtered_correspondence, H, success
    else:
        return None  # a failed matching


# Step 4: stitch the images together
def stitch(img_1, img_2, show_match=False, ratio=0.75, threshold=4):
    """
    stitch the images together

    :param img_1: image 1
    :param img_2: image 2
    :param show_match: if True, show the matching points, otherwise just show the result
    :param ratio: the ratio to be passed to the <code>solve_H()</code> function
    :param threshold: the ratio to be passed to the <code>solve_H()</code> function
    :return: the stitched image
    """
    keypoints_1, descriptor_1 = detectKeypoints(img_1)
    keypoints_2, descriptor_2 = detectKeypoints(img_2)
    corr = solve_H(keypoints_1, descriptor_1, keypoints_2, descriptor_2, ratio, threshold)
    if corr is None:  # matching failed
        return None
    else:
        # get the correspondence
        correspondence, H, success = corr

        # optional: draw the correspondence points
        if show_match:
            draw_correspondence(img_1, keypoints_1, img_2, keypoints_2, correspondence, success)
        # compute the width and the height of the result image
        width = img_1.shape[1] + img_2.shape[1]
        height = img_2.shape[0] + img_1.shape[0]

        # projective transformation using inverse of H
        res = cv2.warpPerspective(img_2, np.linalg.inv(H), (width, height))
        # directly stitch image 1 to the left of the result
        res[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
        return res


# Optional: draw the correspondence of points
def draw_correspondence(img_1, keypoints_1, img_2, keypoints_2, correspondence, success):
    """
    draw the correspondence points

    :param img_1: image 1
    :param keypoints_1: the keypoints of image 1
    :param img_2: image 2
    :param keypoints_2: the keypoints of image 2
    :param correspondence: the correspondence of keypoints
    :param success: the matching status
    :return: no return
    """
    h_1, w_1 = img_1.shape[0:2]
    h_2, w_2 = img_2.shape[0:2]
    # create a blank image. 3 channels, height is maximum of the height of 2 images
    # and the width is the sum of the 2 images
    match_image = np.zeros((max(h_1, h_2), w_1 + w_2, 3), dtype=np.uint8)

    # concatenate the images
    match_image[0:h_1, 0:w_1] = img_1
    match_image[0:h_2, w_1:] = img_2
    successful_point_pairs = 0
    # find the correspondence points and draw lines between them
    for ((idx_2, idx_1), is_success) in zip(correspondence, success):
        # successfully matched
        if is_success == 1:
            successful_point_pairs += 1
            point_1 = (int(keypoints_1[idx_1][0]), int(keypoints_1[idx_1][1]))
            point_2 = (int(keypoints_2[idx_2][0]) + w_1, int(keypoints_2[idx_2][1]))
            cv2.line(match_image, point_1, point_2, (255, 0, 0), 1)
    if successful_point_pairs <= 15:
        print('Failed to stitch the images as too few correspondences are found!')
        exit(-1)
    cv2.imshow('Matching Points', match_image)


if __name__ == '__main__':
    args = get_args().parse_args()
    if len(args.img_name) != 2:
        print(args.img_name)
        raise ValueError('Exactly 2 names of images shall be provided!')
    img_1 = cv2.imread('./img/' + args.img_name[0])
    img_2 = cv2.imread('./img/' + args.img_name[1])
    if args.img_name[0] > args.img_name[1]:
        img_1, img_2 = img_2, img_1
    img_1 = cv2.resize(img_1, (img_1.shape[1] // 2, img_1.shape[0] // 2))
    img_2 = cv2.resize(img_2, (img_2.shape[1] // 2, img_2.shape[0] // 2))
    # cv2.imshow('Image 1', img_1)
    # cv2.imshow('Image 2', img_2)
    result = stitch(img_1, img_2, show_match=args.show_match)
    if result is None:
        print('Failed to stitch the images as too few correspondences are found!')
        exit(-1)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

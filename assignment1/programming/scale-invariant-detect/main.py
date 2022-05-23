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
    parser.add_argument('img_name', default=None, type=str)
    parser.add_argument('--use_cv', default=True, type=str2bool)
    parser.add_argument('--sigma0', default=1.52, type=float)
    return parser


def gaussian(x, y, sigma):
    """
    the gaussian kernel(Really?)

    :param x: number of columns
    :param y: number of rows
    :param sigma: sigma
    :return: the gaussian of the image
    """
    ret = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            ret[i, j] = (np.exp(-(i ** 2 + j ** 2) / (2 * (sigma ** 2)))) / (2 * np.pi * (sigma ** 2))
    return ret


def gaussian_kernel(sigma, dim):
    """
    the gaussian kernel function

    :param sigma: the standard deviation
    :param dim: the dimension of gaussian kernel
    :return: the gaussian kernel
    """
    vec_1dim = [i - dim // 2 for i in range(dim)]  # 1 dimension of the gaussian kernel
    vec = []
    for i in range(dim):
        vec.append(vec_1dim)
    vec = np.array(vec)
    sigma_sq = 2 * sigma * sigma
    return (1. / (np.pi * sigma_sq)) * np.exp(-(vec ** 2 + vec.T ** 2) / sigma_sq)


def convolve(gaussian, img, dim):
    """
    the convolution function

    :param dim: the dimension of the gaussian kernel
    :param gaussian: the gaussian kernel
    :param img: the image
    :return: the convolved matrix
    """

    res = []
    x, y = img.shape

    # pad the image so that edges can be convolved
    img = np.pad(img, ((dim // 2, dim // 2), (dim // 2, dim // 2)), 'constant')

    for i in range(x):
        res.append([])
        for j in range(y):
            res[-1].append((gaussian * img[i:i + dim, j: j + dim]).sum())
    return np.array(res)


# the DoG function
# def DoG(x, y, k, sigma):
#     return gaussian(x, y, k * sigma) - gaussian(x, y, sigma)


def construct_DoG_pyramid(img, sigma0=1.52, S=3, group_factor=3):
    """
    construct gaussian pyramid

    :param group_factor: the group factor to subtract default 3 in Lowe's paper
    :param img: the input image(gray)
    :param sigma0: sigma0 default 1.6 in Lowe's paper
    :param S: the number of layer in an octave of pyramid
    :return: the DoG pyramid
    """
    k = 2 ** (1 / S)
    layers = S + 3
    x, y = img.shape[0:2]
    groups = int(np.log2(min(x, y))) - group_factor
    images = np.zeros(groups + 1, dtype=np.ndarray)
    gaussians = np.zeros((groups + 1, layers), dtype=np.ndarray)
    all_layers = np.zeros_like(gaussians)
    sigmas = np.zeros_like(gaussians)
    DoG = np.zeros((groups + 1, layers - 1), dtype=np.ndarray)

    # sample the image
    for i in range(groups + 1):
        ny = int(y / (2 ** (i - 1)))
        nx = int(x / (2 ** (i - 1)))
        images[i] = cv2.resize(img, (ny, nx))
        # images[i] = img[::(1 << i), ::(1 << i)]

    # compute gaussian pyramid
    for i in range(groups + 1):
        nx, ny = images[i].shape[0:2]
        for j in range(layers):
            sigma = (2 ** i) * (k ** j) * sigma0
            dim = int(6 * sigma + 1)
            dim = dim if dim % 2 != 0 else dim + 1
            # gaussians[i, j] = gaussian(nx, ny, sigma)
            gaussians[i, j] = gaussian_kernel(sigma, dim)
            sigmas[i, j] = sigma

    # apply gaussian pyramid to the sampled images
    for i in range(groups + 1):
        for j in range(layers):
            # all_layers[i, j] = gaussians[i, j] * images[i]
            dim = int(6 * sigmas[i, j] + 1)
            dim = dim if dim % 2 != 0 else dim + 1
            all_layers[i, j] = convolve(gaussians[i, j], images[i], dim)

    # compute DoG
    for i in range(groups + 1):
        for j in range(layers - 1):
            DoG[i, j] = all_layers[i, j + 1] - all_layers[i, j]

    return sigmas[:, 0:5], DoG


def scale_invariant_detect(DoG, sigmas):
    """
    scale invariant detection

    :param DoG: the DoG pyramid
    :param sigmas: the sigma matrix
    :return: the invariant points with its scale
    """
    threshold = 1./255
    invariants = []
    group, layer = DoG.shape[0:2]
    for g in range(group):
        for l in range(1, layer - 1):
            image = DoG[g, l]
            image_upper = DoG[g, l + 1]
            image_lower = DoG[g, l - 1]
            x, y = image.shape[0:2]
            for i in range(1, x - 1):
                for j in range(1, y - 1):
                    # 26 neighbors and the point itself
                    neighbors = [image[i, j], image[i - 1, j], image[i + 1, j],
                                 image[i, j - 1], image[i, j + 1], image[i - 1, j - 1],
                                 image[i - 1, j + 1], image[i + 1, j - 1], image[i + 1, j + 1],

                                 image_upper[i, j], image_upper[i - 1, j], image_upper[i + 1, j],
                                 image_upper[i, j - 1], image_upper[i, j + 1], image_upper[i - 1, j - 1],
                                 image_upper[i - 1, j + 1], image_upper[i + 1, j - 1], image_upper[i + 1, j + 1],

                                 image_lower[i, j], image_lower[i - 1, j], image_lower[i + 1, j],
                                 image_lower[i, j - 1], image_lower[i, j + 1], image_lower[i - 1, j - 1],
                                 image_lower[i - 1, j + 1], image_lower[i + 1, j - 1], image_lower[i + 1, j + 1]
                                 ]
                    if neighbors[0] > threshold and (
                            ((neighbors[0] > 0) and (neighbors[0] > neighbors[1:]).all()) or
                            ((neighbors[0] < 0) and (neighbors[0] < neighbors[1:]).all())):
                        invariants.append((i * (2 ** (g - 1)), j * (2 ** (g - 1)), sigmas[g, l]))
    return invariants


def draw_keypoints(invariants, img):
    """
    draw the scale invariant points

    :param invariants: the invariants point
    :param img: the original image
    :return: the image with scale invariant marked
    """
    img_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(invariants)):
        cv2.circle(img_1, (int(invariants[i][1]), int(invariants[i][0])), int(invariants[i][2]), (0, 0, 255))
    return img_1


if __name__ == '__main__':
    args = get_args().parse_args()
    if args.img_name is None:
        raise ValueError('The name of the image shall be provided!')
    img_name = args.img_name
    img = cv2.imread('./img/' + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use OpenCV API
    if args.use_cv:
        sift = cv2.xfeatures2d.SIFT_create()
        keypoint, descriptor = sift.detectAndCompute(img, None)
        img = cv2.drawKeypoints(image=img, keypoints=keypoint, outImage=img, color=(0, 0, 255), flags=cv2.
                                DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imwrite('./img/' + img_name + '_sid.png', img)
        cv2.imshow(f'Scale Invariant Detection:{img_name}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # use my implementation
    else:

        sigmas, DoG = construct_DoG_pyramid(img, sigma0=args.sigma0)
        invariants = scale_invariant_detect(DoG, sigmas)
        img_1 = draw_keypoints(invariants, img)
        cv2.imshow(f'Scale Invariant Detection:{img_name}', img_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

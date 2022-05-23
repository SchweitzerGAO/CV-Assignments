import numpy as np
from matplotlib import pyplot as plt
import random


def manhattan(y_pred, y_gt):
    """
    computes Manhattan distance of 2 points
    :param y_pred: the predicted y
    :param y_gt: the ground truth of y
    :return: manhattan distance of the prediction and gt
    """
    return np.abs(y_pred - y_gt)


def dist_to_line(k, b, x, y):
    """
    computes distance between a line and a point
    :param k: slope of a line
    :param b: intercept of a line
    :param x: x-coord of a point
    :param y: y-coord of a point
    :return: the distance of a line and a point
    """
    return (np.abs(k * x - y + b)) / (np.sqrt(k * k + 1))


# the RANSAC algorithm
def RANSAC(X, y):
    """
    RANSAC algorithm
    :param X: x-coords of the data
    :param y: y-coords of the data
    :return: best slope and intercept of the 2D line model
    """
    k = 0
    b = 0
    best_inliers = 0
    example_thres = int(X.shape[0] / 1.2)
    dist_thres = 0.1
    iterate = 500  # iteration time, optimized during loop
    best_model_prob = 0.99  # the probability of the expected best model
    for _ in range(iterate):
        inliers = 0
        # sample 2 points for a 2D line
        sample = random.sample(range(X.shape[0]), 2)
        sample_x1 = X[sample[0]]
        sample_y1 = y[sample[0]]
        sample_x2 = X[sample[1]]
        sample_y2 = y[sample[1]]
        # calculate temporary k and b
        k_temp = (sample_y2 - sample_y1) / (sample_x2 - sample_x1)
        b_temp = sample_y1 - k_temp * sample_x1
        for i in range(X.shape[0]):
            if manhattan(k_temp * X[i] + b_temp, y[i]) <= dist_thres:
                inliers += 1
        if inliers > example_thres:
            k = k_temp
            b = b_temp
            break
        else:
            if inliers > best_inliers:
                best_inliers = inliers
                k = k_temp
                b = b_temp
                iterate = np.log(1 - best_model_prob) / np.log(1 - (np.power(inliers / X.shape[0], 2)))

    return k, b


if __name__ == '__main__':
    X = np.array([-2, 0, 2, 3, 4, 5, 6, 8, 10, 12, 13, 14, 16, 18])
    y = np.array([0, 0.9, 2.0, 6.5, 2.9, 8.8, 3.95, 5.03, 5.97, 7.1, 1.2, 8.2, 8.5, 10.1])
    k, b = RANSAC(X, y)
    y_pred = k * X + b
    plt.figure()
    plt.title('RANSAC estimation manhattan')
    plt.xlabel('x value')
    plt.ylabel('y value')
    plt.scatter(X, y, c='r', marker='+')
    plt.plot(X, y_pred)
    text = 'k={:.3f}\nb={:.3f}'.format(k, b)
    plt.text(16, 0, text, fontdict={'size': 8, 'color': 'r'})
    plt.show()

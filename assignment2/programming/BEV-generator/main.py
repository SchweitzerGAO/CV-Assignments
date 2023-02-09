import cv2
import numpy as np
import glob


def camera_calib(h, w):
    path = './data/*.jpg'
    images = glob.glob(path)
    n_image = len(images)
    size = (1280, 1706)
    img_points = []
    print("extracting corners")
    for f_name in images:
        img = cv2.imread(f_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w, h))
        if ret:
            criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.1)
            sub_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            img_points.append(sub_corners)
    print("extracting corners finish")

    print("calibration starts...")
    # print(n_image)
    obj_points = []
    for _ in range(n_image):
        temp_point_set = []
        for i in range(h):
            for j in range(w):
                temp_point_set.append(np.array([i * 15., j * 15., 0.], dtype=np.float32))
        obj_points.append(temp_point_set)
    # print(obj_points)
    calib_criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6)
    ret_val, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array(obj_points), img_points, size, None, None, flags=0,
                                                           criteria=calib_criteria)
    print('calibration ends')
    return ret_val, mtx, dist, rvecs, tvecs


def BEV_generate(w, h):
    size = (1280, 1706)
    img = cv2.imread('img2.jpg')
    # img = cv2.resize(img, size)
    intrinsic_cpp = np.array([[1246.6625, 0, 894.65857],
                              [0, 1245.8951, 665.9408],
                              [0, 0, 1]])
    distortion_cpp = np.array([-0.068792731, -0.14259334, 0.031203667, 0.030004108, -0.28120443])

    intrinsic_py = np.array([[1278.9614, 0, 854.7758],
                             [0, 1279.8275, 648.4584],
                             [0, 0, 1]])
    distortion_py = np.array([0.05246238 - 0.14191777, 0.000993, 0.00201192, 0.15738296])

    img_undistorted_cpp = cv2.undistort(img, intrinsic_cpp, distortion_cpp)
    img_undistorted_cpp_gray = cv2.cvtColor(img_undistorted_cpp, cv2.COLOR_BGR2GRAY)
    # img_undistorted_cpp_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_undistorted_py = cv2.undistort(img, intrinsic_py, distortion_py)
    img_undistorted_py_gray = cv2.cvtColor(img_undistorted_py, cv2.COLOR_BGR2GRAY)

    # the PCS
    ret_cpp, corners_cpp = cv2.findChessboardCorners(img_undistorted_cpp_gray, (w, h))
    ret_py, corners_py = cv2.findChessboardCorners(img_undistorted_py_gray, (w, h))
    # print(ret_cpp, ret_py)
    # the WCS

    # build a WCS
    wcs = np.zeros((w * h, 3), np.float32)
    wcs[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    print(wcs)

    # points of the PCS
    if ret_py and ret_cpp:
        point1 = np.array([corners_cpp[0, :], corners_cpp[8, :], corners_cpp[-9, :], corners_cpp[-1, :]],
                          dtype=np.float32)
    else:
        print("No chessboard corners found!")
        return

    # points of the WCS
    point2 = np.array([wcs[0, :][:-1], wcs[8, :][:-1], wcs[-9, :][:-1], wcs[-1, :][:-1]], dtype=np.float32)
    # print(point2)
    # point2[:] = point2[:] * 20 + 640
    point2[:, 0] = point2[:, 0] * 15 + 400
    point2[:, 1] = point2[:, 1] * 15 + 400
    # print(point2)

    # generate BEV
    M = cv2.getPerspectiveTransform(point1, point2)
    out_img = cv2.warpPerspective(img, M, size)
    cv2.imshow('result', out_img)
    # cv2.imwrite('result.jpg', out_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    h = 6
    w = 9
    # ret_val, mtx, dist, rvecs, tvecs = camera_calib(h, w)
    BEV_generate(w, h)

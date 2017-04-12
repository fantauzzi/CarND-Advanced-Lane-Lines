import cv2
import matplotlib.pyplot as plt
import matplotlib as mplib
import numpy as np
import os
import glob

# Directory containing images for caliration
calibration_dir = 'camera_cal'

# Size of the checkered calibration target
target_size = (9, 6)  # (columns, rows)

output_dir = 'output_images'


def calibrate_camera():
    '''
    Load all images with file name calibration*.jpg from directory `calibration_dir` and use them for camera calibration
    :return: the same values returned by cv2.calibrateCamera(), in the same order 
    '''
    f_names = glob.glob(calibration_dir + '/calibration*.jpg')

    objp = np.zeros((target_size[0] * target_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:target_size[0], 0:target_size[1]].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, f_name in enumerate(f_names):
        img = cv2.imread(f_name)
        assert img is not None
        gscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners TODO try sub-pixel accuracy
        found, corners = cv2.findChessboardCorners(gscale_img, target_size, None)
        if found:
            obj_points.append(objp)
            img_points.append(corners)
        else:
            print("WARNING: couldn't detect calibration pattern in file", f_name)

    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    return ret


def switch_RGB(img):
    '''
    Switches between RGB and BGR representation of a color image, and returns the result
    '''
    return img[:, :, ::-1]


def no_ticks(axes):
    axes.get_xaxis().set_ticks([])
    axes.get_yaxis().set_ticks([])


def save_undistorted_sample(f_name, mtx, dist):
    '''
    Undirstorts the image from the given file based on camera parameters `mtx` and `dist` and saves
    the result in a .png file under `output_dir`, along with the original (distorted) image, for comparison
    '''
    img = cv2.imread(f_name)
    assert img is not None
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    # Switch from BGR to RGB for presentation in Matplotlib
    img = switch_RGB(img)
    undistorted_img = switch_RGB(undistorted_img)

    # Makes the drawing
    fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    axes1.imshow(img)
    axes1.set_title('Original Image', fontsize=30)
    no_ticks(axes1)
    axes2.imshow(undistorted_img)
    axes2.set_title('Undistorted Image', fontsize=30)
    no_ticks(axes2)
    fig.tight_layout()

    # Saves the drawing
    f_basename = os.path.basename(f_name)
    output_f_name = os.path.splitext(f_basename)[0] + '.png'
    fig.savefig(output_dir + '/undistored-' + output_f_name)  # TODO fix all '/' such that it will work under Windows


def save_undistorted(f_name, mtx, dist):
    img = cv2.imread(f_name)
    assert img is not None
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    f_basename = os.path.basename(f_name)
    output_f_name = os.path.splitext(f_basename)[0] + '.png'
    cv2.imwrite(output_dir + '/undistorted-' + output_f_name, undistorted_img)
    plt.close()


def grayscale(img):
    '''
    Converts the given image from BGR to gray-scale and returns the result 
    '''
    gscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return gscale_img[:, :, 0]
    # gscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return gscale_img


def find_line_params(p0, p1):
    '''
    Returns slope and intercept (in that order) of the line going through 2D points `p0` and `p1`.
    '''
    m = (p0[1] - p1[1]) / (p0[0] - p1[0])
    q = p0[1] - m * p0[0]
    return m, q


def find_intersect(p0, p1, p2, p3):
    '''
    Returns the coordinates of the intersection between the line going through `p0` and `p1`, and the line going through
    `p2` and `p3`.
    '''
    m = [0, 0]
    q = [0, 0]
    m[0], q[0] = find_line_params(p0, p1)
    m[1], q[1] = find_line_params(p2, p3)
    x = (q[1] - q[0]) / (m[0] - m[1])
    y = m[0] * x + q[0]
    return x, y


def find_x_given_y(y, p0, p1):
    '''
    returns the value for x for the given `y` along the line passing through points `p0` and `p1`
    '''
    m, q = find_line_params(p0, p1)
    x = (y - q) / m
    return x


def try_intersect():
    a = (0, 0)
    b = (10, 0)
    c = (10, 10)
    i = find_intersect(a, b, a, c)
    assert i == (0, 0)
    d = (20, 0)
    i = find_intersect(a, c, d, c)
    assert i == c
    e = (-10, 10)
    i = find_intersect(c, e, d, c)
    assert i == c
    x = find_x_given_y(5, c, d)
    assert x == 15


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()

    # Save an image with one calibration sample along with its undistorted version
    save_undistorted_sample(calibration_dir + '/calibration2.jpg', mtx, dist)

    #############################################################
    test_image = 'test_images/straight_lines2.jpg'
    # test_image = 'test_images/test5.jpg'
    save_undistorted(test_image, mtx, dist)

    # Load a test image (for now)
    img = cv2.imread(test_image)
    assert img is not None

    # Un-distort it applying camera calibration
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    # Convert to the gray-scale of choice
    # gscale_img = grayscale(undistorted_img)

    lane_l = ((264, 688), (621, 431))
    lane_r = ((660, 431), (1059, 688))
    v_point = find_intersect(*lane_l, *lane_r)
    target_shape = (undistorted_img.shape[0], undistorted_img.shape[1])
    # y1, y2 = round(.2*target_shape[0]), target_shape[0]-1
    # y1, y2 = 431, 688
    y1, y2 = round((target_shape[0] - v_point[1]) * .13 + v_point[1]), target_shape[0] - 51
    assert v_point[1] <= y1 <= target_shape[0] - 1
    assert v_point[1] <= y2 <= target_shape[0] - 1
    source = np.float32([
        [find_x_given_y(y2, *lane_l), y2],
        [find_x_given_y(y1, *lane_l), y1],
        [find_x_given_y(y1, *lane_r), y1],
        [find_x_given_y(y2, *lane_r), y2]
    ])

    # TODO compute the vanishing point of the lane lines, and use lines from there to the lower corners of the image
    '''source = np.float32([[264, 688],
                         [621, 431],
                         [660, 431],
                         [1059, 688]])'''
    # target_size = (undistorted_img.shape[1], undistorted_img.shape[0])  # (No. of columns, No. of rows)
    # TODO remove hard-wiring of numbers below
    target = np.float32([[source[0, 0], target_shape[0] - 1],
                         [source[0, 0], 0],
                         [source[3, 0], 0],
                         [source[3, 0], target_shape[0] - 1]])
    source = np.expand_dims(source, 1)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src=source, dst=target)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undistorted_img, M, target_shape[::-1])

    gscale = grayscale(warped)

    sobel_x = cv2.Sobel(gscale, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gscale, cv2.CV_64F, 0, 1, ksize=3)

    grad_size = (sobel_x ** 2 + sobel_y ** 2) ** .5
    max_grad_size = np.max(grad_size)
    grad_size = np.uint8(grad_size / max_grad_size * 255)
    grad_dir = np.arctan2(sobel_y, sobel_x)

    # sobel_8u = np.uint8(abs_sobel64f)

    plt.figure()
    plt.imshow(grad_size, cmap='gray')
    # plt.imshow(switch_RGB(warped))
    plt.show()


if __name__ == '__main__':
    main()

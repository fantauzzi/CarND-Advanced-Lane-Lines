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
        # Find the chessboard corners
        found, corners = cv2.findChessboardCorners(gscale_img, target_size, None)
        if found:
            obj_points.append(objp)
            img_points.append(corners)
        else:
            print("WARNING: couldn't find calibration pattern in file ", f_name)

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
    the result in a .png file under `output_dir`
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
    fig.savefig(output_dir + '/undistored-' + output_f_name)


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()

    # Save an image with one calibration sample along with its undistorted version
    save_undistorted_sample(calibration_dir + '/calibration2.jpg', mtx, dist)


if __name__ == '__main__':
    main()

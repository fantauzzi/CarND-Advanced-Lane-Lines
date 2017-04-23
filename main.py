import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import math
import time
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons
from scipy.signal import gaussian
from enum import IntEnum


def switch_RGB(img):
    '''
    Switches between RGB and BGR representation of a color image, and returns the result
    '''
    return img[:, :, ::-1]


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
    Returns the value for x for the given `y` along the line passing through points `p0` and `p1`
    '''
    m, q = find_line_params(p0, p1)
    x = (y - q) / m
    return x


def calibrate_camera(calibration_dir, target_size, print_error=True):
    """
    Calibrate a camera using images of the checkered pattern taken with the camera. 
    :param calibration_dir: the directory containing the input images.  
    :param target_size: the size of the checkered pattern, a pair as (No_of_columns, No_of_rows).
    :return: the calibration parameters as returned by cv2.calibrateCamera()
    """

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
            print("WARNING: couldn't detect calibration pattern in file", f_name)

    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    if print_error:
        mean_error = 0
        for i in range(len(obj_points)):
            img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error
            assert error >= 0
        print("Measured calibration error: ", mean_error / len(obj_points))

    return new_mtx, dist, rvecs, tvecs, roi


def find_gradient(gscale_image):
    """
    Returns the gradient modulus and direction absolute value for the given grayscale image.
    Modulus is scaled to be in the range of integers [0, 255], direction to be in the real numbers interval
    [0, Pi]
    """
    sobel_x = cv2.Sobel(gscale_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gscale_image, cv2.CV_64F, 0, 1, ksize=5)
    grad_size = (sobel_x ** 2 + sobel_y ** 2) ** .5
    max_grad_size = np.max(grad_size)
    grad_size = np.uint8(grad_size / max_grad_size * 255)
    grad_dir = np.abs(np.arctan2(sobel_y, sobel_x))
    return grad_size, grad_dir


def find_x_gradient(gscale_image):
    """
    Returns the absolute value of the x-gradient for the given grayscale image, scaled to be in the range of integers 
    [0, 255].
    """
    sobel_x = cv2.Sobel(gscale_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_x = np.abs(sobel_x)
    max_grad_x = np.max(sobel_x)
    sobel_x = np.uint8(sobel_x / max_grad_x * 255)
    return sobel_x


def undistort_image(image, mtx, dist, roi):
    """
    Returns an undistorted copy of the given image, based on the given camera calilbration parameters
    """
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
    # x, y, w, h = roi
    # undistorted_image= undistorted_image[y:y + h, x:x + w]
    return undistorted_image


class SlidingWindow:
    def __init__(x0, y0, width, height):
        pass


class LaneLine:
    def __init__(self, windows_shape, image_shape):
        self._windows_shape = windows_shape
        self._image_shape = image_shape
        assert image_shape[0] % windows_shape[0] == 0
        self._n_bands = image_shape[0] // windows_shape[0]
        self._centroids = np.array([None] * self._n_bands)
        self._bottom_x = None

    def get_centroid_x(self, band):
        return self._centroids[band]

    def get_bottom_x(self):
        return self._bottom_x

    def get_recenter_roi(self, _):
        raise NotImplementedError

    def get_printable_name(self):
        raise NotImplementedError

    def recenter(self, thresholded, filter):
        assert thresholded.shape == self._image_shape
        print('Re-centering '+self.get_printable_name())
        index, offset = self.get_recenter_roi(thresholded)
        area_sum = np.sum(thresholded[index], axis=0)
        self._bottom_x = np.argmax(np.convolve(filter, area_sum)) - self._windows_shape[1] / 2 + offset
        return self._bottom_x


class LeftLaneLine(LaneLine):
    #def __init__(self):
    #    pass

    def get_printable_name(self):
        return 'left'

    def get_recenter_roi(self, thresholded):
        index = np.s_[int(3 * thresholded.shape[0] / 4):, :int(thresholded.shape[1] / 2)]
        offset = 0
        return index, offset


class RightLaneLine(LaneLine):
    #def __init__(self):
    #    pass

    def get_printable_name(self):
        return 'right'

    def get_recenter_roi(self, thresholded):
        index = np.s_[int(3 * thresholded.shape[0] / 4):, int(thresholded.shape[1] / 2):]
        offset = int(thresholded.shape[1] / 2)
        return index, offset


class ImageProcessing:
    def __init__(self):
        self._unprocessed = None
        self.invalidate()
        # Computation parameters, tune with care
        # TODO give them beter names and document them
        self.min_acceptable = 10.
        self.centroid_window_width = 100
        self.centroid_window_height = 80
        self.centroid_window_margin = 50
        self.filter = gaussian(self.centroid_window_width, std=self.centroid_window_width / 8, sym=True)

    def invalidate(self):
        self._top_view = None
        self._thresholded = None
        self._lane_lines = None

    def get_top_view(self):
        if self._top_view is None:
            assert self._unprocessed is not None
            image = self._unprocessed
            target_shape = (image.shape[0], image.shape[1])  # (rows, columns)

            # Begin by finding the perspective vanishing point
            lane_l = ((264, 688), (621, 431))  # Two points identifying the line going along the left lane marking
            lane_r = ((660, 431), (1059, 688))  # Two points identifying the line going along right lane marking
            v_point = find_intersect(*lane_l,
                                     *lane_r)  # Intersection of the two lines above (perspective vanishing point)

            # Alternative approach, commented out below, with lines going from vanishing points to lower image corners
            '''lane_l2 = (
            (0, target_shape[0] - 1), v_point)  # Two points identifying the line going along the left lane marking
            lane_r2 = ((target_shape[1] - 1, target_shape[0] - 1),
                       (v_point))  # Two points identifying the line going along right lane marking'''

            # Determine a quadrangle in the source image
            source_h = image.shape[0]
            y1, y2 = round((source_h - v_point[1]) * .13 + v_point[1]), source_h - 51
            assert v_point[1] <= y1 <= source_h - 1
            assert v_point[1] <= y2 <= source_h - 1
            source = np.float32([
                [find_x_given_y(y2, *lane_l), y2],
                [find_x_given_y(y1, *lane_l), y1],
                [find_x_given_y(y1, *lane_r), y1],
                [find_x_given_y(y2, *lane_r), y2]
            ])

            # Determine the corresponing quandrangle in the target (destination) image
            target = np.float32([[source[0, 0], target_shape[0] - 1],
                                 [source[0, 0], 0],
                                 [source[3, 0], 0],
                                 [source[3, 0], target_shape[0] - 1]])

            # Alternative approach, commented out below
            '''target = np.float32([[0, target_shape[0] - 1],
                                 [0, 0],
                                 [target_shape[1] - 1, 0],
                                 [target_shape[1] - 1, target_shape[0] - 1]])'''

            # Given the source and target quadrangles, calculate the perspective transform matrix
            source = np.expand_dims(source, 1)  # OpenCV requires this extra dimension
            M = cv2.getPerspectiveTransform(src=source, dst=target)

            self._top_view = cv2.warpPerspective(image, M, image.shape[1::-1])
        return self._top_view

    def get_thresholded(self):
        if self._thresholded is None:
            assert self._top_view is not None
            # TODO optimise the masks
            masks = (((0, 100, 100), (50, 255, 255)),
                     ((18, 0, 180), (255, 80, 255)),
                     ((4, 0, 180), (15, 80, 255)))
            min_grad_size = 15
            hsv = cv2.cvtColor(self._top_view, cv2.COLOR_BGR2HSV)
            thresholded = np.zeros_like(hsv[:, :, 0])
            MIN, MAX = 0, 1
            H, S, V = 0, 1, 2
            grad_size = find_x_gradient(hsv[:, :, 2])
            for mask in masks:
                thresholded[(mask[MIN][H] <= hsv[:, :, H]) &
                            (mask[MAX][H] >= hsv[:, :, H]) &
                            (mask[MIN][S] <= hsv[:, :, S]) &
                            (mask[MAX][S] >= hsv[:, :, S]) &
                            (mask[MIN][V] <= hsv[:, :, V]) &
                            (mask[MAX][V] >= hsv[:, :, V]) &
                            (grad_size >= min_grad_size)] = 255
            self._thresholded = thresholded
        return self._thresholded

    def position_windows(self):
        """
        Determines the locations in the thresholded images that should be occoupied by either lane line marks (left
         and right), as a set of windows. Instantiates two LaneLine objects and store them in _lane_lines, to hold
         each information about the respective windows.
        """
        assert self._thresholded is not None
        args = ((self.centroid_window_height, self.centroid_window_width), self._thresholded.shape)
        self._lane_lines = [LeftLaneLine(*args), RightLaneLine(*args)]
        assert self._thresholded.shape[0] % self.centroid_window_height == 0
        n_bands = self._thresholded.shape[0] // self.centroid_window_height
        ''' Partition the image in horizontal bands of height self.height, numbered starting from 0, where band 0
        is at the bottom of the image (closest to the camera) '''
        convolved_bands = []
        for band in range(n_bands):
            # convolve the band with a pre-computed filter, stored in self.filter, to detect lane line markers
            image_band = np.sum(
                self._thresholded[
                int(self._thresholded.shape[0] - (band + 1) * self.centroid_window_height):int(
                    self._thresholded.shape[0] - band * self.centroid_window_height),
                :],
                axis=0)
            convolved_bands.append(np.convolve(self.filter, image_band))

        new_centroids_x = [[], []]  # Will collect two lists of centroid x coordinates, one list per lane line
        for lane_line in self._lane_lines:
            ''' For every band in the thresholded image starting from the bottom, find a window in that band containing
            lane line points '''
            lane_centroids_x = []  # Will collect x coordinate of centroids for the current lane_line, one per band
            for band in range(n_bands):
                ''' First determine a `staring_x` value, in which surroundings to look for lane line points for the
                        current band. Then look in a sliding window in the current band around `starting_x`'''
                if band == 0:
                    ''' Special treatement for band 0, at the bottom of the image. If you don't already have a window
                    in that band where to look for lane lines (from previous frames), then determine it staring from the
                    hystogram of the left or right bottom quarter of the image (left or right, depending on which lane). '''
                    if lane_line.get_centroid_x(0) is None:
                        print('Recentering lane')
                        lane_line.recenter(self._thresholded, self.filter)
                    starting_x = lane_line.get_bottom_x()
                else:
                    ''' For other bands different from the bottom one, find the first window (from the top) in a band below,
                    and use the x of its centroid as teh starting x for the window'''
                    for band_below in range(band - 1, -1, -1):
                        if lane_centroids_x[band_below] is not None:
                            starting_x = lane_centroids_x[band_below]
                            break
                    else:
                        starting_x = lane_line.get_bottom_x()
                        assert starting_x is not None
                ''' Now that you have `starting_x`, do a convolution of the thresholded image, with a filter, around
                `starting_x` in the current band, looking for lane line points '''
                offset = self.centroid_window_width / 2
                min_index = int(max(starting_x + offset - self.centroid_window_margin, 0))
                max_index = int(min(starting_x + offset + self.centroid_window_margin, self._thresholded.shape[1]))
                # Compute the x coordinate of the centroid of the window that contains lane line points
                centroid_x = np.argmax(convolved_bands[band][min_index:max_index]) + min_index - offset
                # TODO test for goodness
                goodness = np.max(convolved_bands[band][min_index:max_index])
                # Update the list of centroid x coordinates with what just found for the current lane line and band
                lane_centroids_x.append(centroid_x)
            new_centroids_x.append(lane_centroids_x)
            # Now verify the sanity of found centroids, and update the two lane lines with centroids that passed the sanity test

    def get_lanes(self):
        pass

    def process_frame(self, frame):
        self._unprocessed = frame
        self.invalidate()
        top_view = self.get_top_view()
        thresholded = self.get_thresholded()
        self.position_windows()
        thresholded_color = cv2.merge((thresholded, thresholded, thresholded))
        return thresholded_color


if __name__ == '__main__':
    # input_fname = 'project_video.mp4'
    input_fname = 'challenge_video.mp4'
    output_dir = 'output_images'  # TODO use command line parameters instead
    # Directory containing images for caliration
    calibration_dir = 'camera_cal'
    # Size of the checkered calibration target
    target_size = (9, 6)  # (columns, rows)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calibrate the camera
    mtx, dist, rvecs, tvecs, roi = calibrate_camera(calibration_dir, target_size)

    # Save an image with one calibration sample along with its undistorted version
    # save_undistorted_sample(calibration_dir + '/calibration2.jpg', mtx, dist, roi)

    output_fname = 'out_' + input_fname
    vidcap = cv2.VideoCapture(input_fname)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vertical_resolution = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert fps > 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidwrite = cv2.VideoWriter(output_fname, fourcc=fourcc, fps=fps, frameSize=(1280, 720))
    print('Source video', input_fname, 'is at', fps, 'fps with vertical resolution of', int(vertical_resolution),
          'pixels')

    frame_counter = 0
    # vidcap.set(cv2.CAP_PROP_POS_MSEC, 6000)
    start_time = time.time()

    processor = ImageProcessing()
    while (True):  # TODO consider to move it to an OpenCV loop
        read, frame = vidcap.read()
        if not read:
            break
        frame_counter += 1
        print('Processing frame', frame_counter)
        # Un-distort it applying camera calibration
        undistorted_img = undistort_image(frame, mtx, dist, roi)
        processed = processor.process_frame(undistorted_img)
        vidwrite.write(processed)
        if frame_counter % 100 == 0:
            pass

    print('\nProcessing rate {:.1f} fps'.format(frame_counter / (time.time() - start_time)))

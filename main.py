import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import math
import time
import copy
import argparse
import PIL
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons
from scipy.signal import gaussian
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def switch_RGB(img):
    """
    Switches between RGB and BGR representation of a color image, and returns the result
    """
    return img[:, :, ::-1]


def find_line_params(p0, p1):
    """
    Returns slope and intercept (in that order) of the line going through 2D points `p0` and `p1`.
    """
    m = (p0[1] - p1[1]) / (p0[0] - p1[0])
    q = p0[1] - m * p0[0]
    return m, q


def find_intersect(p0, p1, p2, p3):
    """
    Returns the coordinates of the intersection between the line going through `p0` and `p1`, and the line going through
    `p2` and `p3`.
    """
    m = [0, 0]
    q = [0, 0]
    m[0], q[0] = find_line_params(p0, p1)
    m[1], q[1] = find_line_params(p2, p3)
    x = (q[1] - q[0]) / (m[0] - m[1])
    y = m[0] * x + q[0]
    return x, y


def find_x_given_y(y, p0, p1):
    """
    Returns the x value for the given `y` along the line passing through points `p0` and `p1`
    """
    m, q = find_line_params(p0, p1)
    x = (y - q) / m
    return x


def calibrate_camera(calibration_dir, target_size, print_error=True):
    """
    Calibrate a camera using images of the checkered pattern taken with the camera. 
    :param calibration_dir: the directory containing the input images in JPEG format.  
    :param target_size: the size of the checkered pattern, a pair as in (No_of_columns, No_of_rows).
    :param print_error: if set to True, after calibration the function evaluates and prints its mean error on the input images
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
        print("Measured calibration mean error: ", mean_error / len(obj_points))

    return new_mtx, dist, rvecs, tvecs, roi


def find_gradient(gscale_image):
    """
    Returns the gradient modulus and direction in absolute value for the given grayscale image.
    Modulus is scaled to be in the range of integers [0, 255]; direction is in the real numbers interval
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


def undistort_image(image, mtx, dist, roi=None):
    """
    Returns an undistorted copy of the given image, based on the given camera calilbration parameters
    """
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
    if roi is not None:  # Clipping
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y + h, x:x + w]
    return undistorted_image


class Centroid:
    _min_goodness = 1

    def __init__(self, x, goodness):
        self.x = x
        self.goodness = goodness

    def is_good(self):
        return self.goodness >= Centroid._min_goodness


def window_mask(width, height, img_ref, center, band):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (band + 1) * height):int(img_ref.shape[0] - band * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def measure_curvature(coefficients, y0, mx, my):
    """
    Finds the curvature radius and centre of curvature, in meters, for the given parabola, at the given point.
    :param coefficients: the parabola coefficients, i.e. [a, b, c] where x=a*y**2+b*y+c. The parabola is assumed
    to be in a reference system in pixels.
    :param y0: the y coordinate, in pixels, of the point along the parabola where the curvature is to be computed.
    :param mx: scale for the x-axis, in meters/pixel (i.e. one pixel along the x-axis corresponds to `mx` meter).
    :param my: scale for the y-axis, in meters/pixel (i.e. one pixel along the x-axis corresponds to `my` meter).
    :return: the curvature radius in meters.
    """

    a = mx / (my ** 2) * coefficients[0]
    b = mx / my * coefficients[1]
    c = coefficients[2]
    Y0 = y0 * my

    radius = ((1 + (2 * a * Y0 + b) ** 2) ** 1.5) / (2 * a)

    return radius


class LaneLine:
    _smoothing = .5

    def __init__(self, windows_shape, image_shape, scale):
        self._windows_shape = windows_shape
        self._image_shape = image_shape
        assert image_shape[0] % windows_shape[0] == 0
        self._n_bands = image_shape[0] // windows_shape[0]
        self._scale = scale  # [mx, my]
        self._centroids = np.array([None] * self._n_bands)
        self._bottom_x = None
        self._coefficients = None
        self._smoothing_coefficients = None
        self._curvature_center = None
        self._is_unreliable = None
        self._fit_undone = None

    def get_coefficients(self):
        return tuple(self._coefficients)

    def get_centroid(self, band):
        return self._centroids[band]

    def get_centroids(self):
        return copy.deepcopy(self._centroids)

    def get_bottom_x(self):
        return self._bottom_x

    def get_recenter_roi(self, _):
        raise NotImplementedError

    def get_printable_name(self):
        raise NotImplementedError

    def is_unreliable(self):
        return self._is_unreliable

    def mark_unreliable(self):
        self._is_unreliable = True

    def get_curvature(self):
        return self._curvature_radius

    def recenter(self, thresholded, filter):
        assert thresholded.shape == self._image_shape
        index, offset = self.get_recenter_roi(thresholded)
        area_sum = np.sum(thresholded[index], axis=0)
        convolution = np.convolve(filter, area_sum, mode='same')
        if np.max(convolution) > 0:
            self._bottom_x = np.argmax(convolution) + offset
        elif self._bottom_x is None:
            self._bottom_x = offset + self._image_shape[1] / 4
        return self._bottom_x

    def set_centroids(self, centroids):
        self._centroids = copy.deepcopy(centroids)

    def check_reliability(self):
        """
        Checks the reliability of the interpolated lane line, stores the result in `self._is_unreliable` as a boolean
        and returns it.
        """
        if abs(self._curvature_radius) < 50:
            self._is_unreliable = True
            print(1)
        good_centroids_count = sum(centroid.is_good() for centroid in self._centroids)
        if good_centroids_count < 2:
            self._is_unreliable = True
            print(2)
        return self._is_unreliable

    def check_reliability_against(self, lane_line):
        """
        Check the reliability of two interpolated lane lines against each other, i.e. verify if they are consistent
        with each other. The `self` lane line is checked against the given one. Returns the result as a boolean. 
        """

        # Fetch the curavture radius and its absolute value for both lane lines
        radius1 = self._curvature_radius
        radius2 = lane_line._curvature_radius
        abs_radius1 = abs(radius1)
        abs_radius2 = abs(radius2)

        straight = 3000
        bent = 620

        def mark_both_unreliable():
            self.mark_unreliable()
            lane_line.mark_unreliable()

        ''' If one is bent and the other is straight, mark them as unreliable'''
        if (abs_radius1 > straight and abs_radius2 < bent) or (abs_radius2 > straight and abs_radius1 < bent):
            mark_both_unreliable()
            return

        ''' Extremely unlikely, but prevent division by zero later on '''
        if radius1 == 0 or radius2 == 0:
            mark_both_unreliable()
            return

        ''' If both are bent but with opposing curvature, mark them as unreliable'''
        if abs_radius1 < bent and abs_radius2 < bent and radius1 / radius2 < 0:
            mark_both_unreliable()
            return

        ''' If the ratio between the two curvature radii (in absolute value) is too low or high, mark both
        lane lines as unreliable'''

        abs_ratio = abs_radius1 / abs_radius2
        if abs_ratio > 5 or abs_ratio < 1 / 5:
            mark_both_unreliable()
            return

        ''' If a lane line falls outside of its half-image at the bottom, mark it as unreliable'''
        x1, x2 = get_lane_lines_position_at(self, lane_line, self._image_shape[0] - 1)
        x_left = min(x1, x2)
        x_right = max(x1, x2)
        lane_left = self if x1 < x2 else lane_line
        lane_right = self if x1 >= x2 else lane_line
        if x_left < 0 or x_left > self._image_shape[1] // 2:
            lane_left.mark_unreliable()
        if x_right <= self._image_shape[1] // 2 or x_right >= self._image_shape[1]:
            lane_right.mark_unreliable()

        ''' If both lane lines are bent, then their curvature radius cannot be too different '''
        if abs_radius1 < bent and abs_radius2 < bent and (abs_ratio > 1.59 or abs_ratio < 1 / 1.59):
            mark_both_unreliable()
            return

        ''' If lane width is too small, mark both lane lines as unreliable'''
        if abs(x1 - x2) * self._scale[0] < 2.5:
            mark_both_unreliable()
            return

    def fit(self, thresholded):
        """
        Interpolates the points in `thresholded` that are believed to belong to the lane line,
        based on current `_centroids`, with a parabola; smooths the parabola with those previously found, and stores
        its coefficients in `_coefficients`.
        """

        def fit_RANSAC(x, Y):
            """
            Fits the given points with a parabola using RANSAC, and returns its coefficients [a, b, c], where a is 
            the highest degree coefficient, c is the lowest.
            """
            estimator = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(), min_samples=2,
                                                     random_state=2111970)
            # The Scikit pipeline is not strictly necessary for least means-square linear interpolation, but it makes it
            # easier to replace the estimator used, and change the degree of the polynomial approximation
            model = make_pipeline(PolynomialFeatures(2), estimator)
            model.fit(x.reshape(-1, 1), Y)
            y0 = model.predict([[-1]])[0]
            y1 = model.predict([[0]])[0]
            y2 = model.predict([[1]])[0]
            a = y0 + (y2 - y0) / 2 - y1
            b = (y2 - y0) / 2
            c = y1
            return np.array([a, b, c])

        def fit_least_square(x, Y):
            """
            Fits the given points with a parabola using least-square interpolation and returns its coefficients.
            """
            coefficients = np.polyfit(x, Y, 2)
            return coefficients

        lane_points = np.zeros_like(thresholded)

        # Go through each band and draw into `lane_points` all points from `thresholded` that are in any sliding window
        for band, centroid in enumerate(self._centroids):
            if centroid.is_good():
                mask = window_mask(self._windows_shape[1], self._windows_shape[0], thresholded, centroid.x, band)
                lane_points[(mask == 1) & (thresholded == 255)] = 255

        # Fit points believed to belong to lane line markers by interpolation
        point_coords = np.where(lane_points == 255)
        if len(point_coords[0]) > 0:
            coefficients = fit_least_square(point_coords[0], point_coords[1])
            # Do the smoothing
            if self._smoothing_coefficients is None:
                self._smoothing_coefficients = coefficients
                self._coefficients = coefficients
            else:
                self._coefficients = (1 - LaneLine._smoothing) * coefficients + LaneLine._smoothing * \
                                                                                self._smoothing_coefficients
                # new_smoothing_coefficients[side] = self.coefficients[side]
                self._smoothing_coefficients = coefficients
            # Measure and store the curvature radius
            self._curvature_radius = measure_curvature(self._coefficients,
                                                       thresholded.shape[0] - 1, self._scale[0],
                                                       self._scale[1])
            self._is_unreliable = False
            self._fit_undone = False
            self.check_reliability()


class LeftLaneLine(LaneLine):
    def get_printable_name(self):
        return 'left'

    def get_recenter_roi(self, thresholded):
        index = np.s_[int(3 * thresholded.shape[0] / 4):, :int(thresholded.shape[1] / 2)]
        offset = 0
        return index, offset


class RightLaneLine(LaneLine):
    def get_printable_name(self):
        return 'right'

    def get_recenter_roi(self, thresholded):
        index = np.s_[int(3 * thresholded.shape[0] / 4):, int(thresholded.shape[1] / 2):]
        offset = int(thresholded.shape[1] / 2)
        return index, offset


def get_lane_width(lane_line1, lane_line2, y):
    """
    Returns the distance in pixels between the two lane lines as displayed in a top view, taken horizontally
    at the given y coordinate. If either lane line was not actually found, returns None.
    """
    x1, x2 = get_lane_lines_position_at(lane_line1, lane_line2, y)
    if x1 is None or x2 is None:
        return None
    return abs(x1 - x2)


def get_lane_lines_position_at(lane_line1, lane_line2, y):
    """
    Returns the x values of the interpolated lane lines for the given `y`. If no lane line was actually interpolated,
    returns (None, None).
    """
    coefficients1 = lane_line1.get_coefficients()
    if coefficients1 is None:
        return None, None
    coefficients2 = lane_line2.get_coefficients()
    if coefficients2 is None:
        return None, None

    # Could do it fancier, with vector calculus
    x1 = coefficients1[0] * (y ** 2) + coefficients1[1] * y + coefficients1[2]
    x2 = coefficients2[0] * (y ** 2) + coefficients2[1] * y + coefficients2[2]
    return x1, x2


def turn_grayscale_into_color(image):
    assert len(image.shape) == 2
    color_image = cv2.merge((image, image, image))
    return color_image


class ImageProcessing:
    def __init__(self):
        self._unprocessed = None
        self._lane_lines = None
        self._prev_lane_lines = None
        self._plot_y = None
        self._M = None
        self.invalidate()
        # Computation parameters, tune with care
        # measurement unit conversion rate for the top view in the x direction, in meters per pixel
        self._mx = 3.66 / 748
        # measurement unit conversion rate for the top view in the y direction, in meters per pixel
        self._my = 3.48 / 93
        # Parameters coverning the size of the sliding window to find lane lines, and how much it slides either way
        self._sliding_window_width = 100
        self._sliding_window_height = 80
        self._sliding_window_margin = 100
        # Filter used for convolution to find the lane line within a sliding window
        self._filter = gaussian(self._sliding_window_width, std=self._sliding_window_width / 8, sym=True)
        # Font used for image overlay
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def invalidate(self):
        # Top-view of the image
        self._top_view = None
        # Result of thresholding the top-view
        self._thresholded = None
        # Points from the top-view believed to belong to either lane line
        self._lanes_points = None

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
            self._M = cv2.getPerspectiveTransform(src=source, dst=target)

            self._top_view = cv2.warpPerspective(image, self._M, image.shape[1::-1])
        return self._top_view

    def get_thresholded(self):
        if self._thresholded is None:
            assert self._top_view is not None
            masks = (((0, 100, 100), (50, 255, 255)),
                     ((18, 0, 180), (255, 80, 255)),
                     ((4, 0, 180), (15, 80, 255)),
                     ((15, 15, 100), (25, 150, 255)))

            min_grad_size = 10
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
        if self._lane_lines is None:
            args = (
                (self._sliding_window_height, self._sliding_window_width), self._thresholded.shape,
                (self._mx, self._my))
            self._lane_lines = [LeftLaneLine(*args), RightLaneLine(*args)]
        assert self._thresholded.shape[0] % self._sliding_window_height == 0
        n_bands = self._thresholded.shape[0] // self._sliding_window_height

        ''' Partition the image in horizontal bands of height self.height, numbered starting from 0, where band 0
        is at the bottom of the image (closest to the camera) '''
        convolved_bands = []
        for band in range(n_bands):
            # convolve the band with a pre-computed filter, stored in self._filter, to detect lane line markers
            image_band = np.sum(
                self._thresholded[
                int(self._thresholded.shape[0] - (band + 1) * self._sliding_window_height):int(
                    self._thresholded.shape[0] - band * self._sliding_window_height),
                :],
                axis=0)
            convolved_bands.append(np.convolve(self._filter, image_band, mode='same'))

        new_centroids_x = []  # Will collect two lists of centroid x coordinates, one list per lane line
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
                    if lane_line.get_centroid(0) is None or not lane_line.get_centroid(0).is_good():
                        print('Recentering lane', lane_line.get_printable_name())
                        lane_line.recenter(self._thresholded, self._filter)
                    starting_x = lane_line.get_bottom_x()
                else:
                    ''' For other bands different from the bottom one, find the first window (from the top) in a band below,
                    and use the x of its centroid as teh starting x for the window'''
                    for band_below in range(band - 1, -1, -1):
                        if lane_centroids_x[band_below] is not None and lane_centroids_x[band_below].is_good():
                            starting_x = lane_centroids_x[band_below].x
                            break
                    else:
                        starting_x = lane_line.get_bottom_x()
                        assert starting_x is not None
                ''' Now that you have `starting_x`, do a convolution of the thresholded image, with a filter, around
                `starting_x` in the current band, looking for lane line points '''
                min_index = int(max(starting_x - self._sliding_window_margin, 0))
                max_index = int(min(starting_x + self._sliding_window_margin, self._thresholded.shape[1]))
                # Compute the x coordinate of the centroid of the window that contains lane line points
                # centroid_x = np.argmax(convolved_bands[band][min_index:max_index]) + min_index - offset
                centroid_x = np.argmax(convolved_bands[band][min_index:max_index]) + min_index
                goodness = np.sum(convolved_bands[band][min_index:max_index])
                # Update the list of centroid x coordinates with what just found for the current lane line and band
                lane_centroids_x.append(Centroid(centroid_x, goodness))
            new_centroids_x.append(lane_centroids_x)

        for lane_line, centroids in zip(self._lane_lines, new_centroids_x):
            lane_line.set_centroids(centroids)

    def fit_lane_lines(self):
        # Interpolate the points believed to belong to either lane lilne
        for lane_line in self._lane_lines:
            lane_line.fit(self._thresholded)

        ''' If either interpolated lane line is found to be unreliable, then replace it with the lane line
        interpolated in the previous frame '''
        if self._prev_lane_lines is None:
            self._prev_lane_lines = []
            for lane_line in self._lane_lines:
                self._prev_lane_lines.append(copy.deepcopy(lane_line))
        else:
            for i_lane, lane_line in enumerate(self._lane_lines):
                if lane_line._is_unreliable:
                    self._lane_lines[i_lane] = copy.deepcopy(self._prev_lane_lines[i_lane])  # !!
                    self._lane_lines[i_lane].mark_unreliable()  # Make sure the lane line is still marked as unreliable

        # Now check the consistency of the two lane lines against each other
        self._lane_lines[0].check_reliability_against(self._lane_lines[1])

        for i_lane, lane_line in enumerate(self._lane_lines):
            if lane_line._is_unreliable:
                self._lane_lines[i_lane] = copy.deepcopy(self._prev_lane_lines[i_lane])  # !!
                self._lane_lines[i_lane].mark_unreliable()
            else:
                self._prev_lane_lines[i_lane] = copy.deepcopy(lane_line)

    def overlay_windows(self, image):
        # Draw the sliding windows
        image_with_overlay = None
        for lane_line in self._lane_lines:
            centroids = lane_line.get_centroids()
            for band, centroid in enumerate(centroids):
                if centroid is not None:
                    rect_x0 = int(centroid.x) - self._sliding_window_width // 2
                    rect_y0 = self._sliding_window_height * (len(centroids) - band) - 1
                    rect_color = (0, 255, 0) if centroid.is_good() else (0, 0, 255)
                    image_with_overlay = cv2.rectangle(image,
                                                       (rect_x0, rect_y0),
                                                       (rect_x0 + self._sliding_window_width,
                                                        rect_y0 - self._sliding_window_height),
                                                       color=rect_color)
                    text_color = (255, 255, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    gap_x = 10
                    gap_y = 30
                    cv2.putText(image_with_overlay, "{:.1f}".format(centroid.goodness),
                                (rect_x0 + self._sliding_window_width + gap_x, rect_y0 - gap_y), font, .5, text_color,
                                1,
                                cv2.LINE_AA)

        return image_with_overlay if image_with_overlay is not None else image

    def get_lanes_points(self, v_res):
        fit_x = [None, None]
        if self._lanes_points is None:
            if self._plot_y is None:
                self._plot_y = np.linspace(0, v_res - 1, v_res)
            for lane_i, lane_line in enumerate(self._lane_lines):
                coefficients = lane_line.get_coefficients()
                if coefficients is None:
                    continue
                fit_x[lane_i] = coefficients[0] * self._plot_y ** 2 + coefficients[1] * self._plot_y + coefficients[2]
        assert self._plot_y is not None
        return self._plot_y, fit_x

    def overlay_lane_lines(self, image):
        assert self._lane_lines is not None

        image_with_lane_lines = image
        plot_y, lanes_points = self.get_lanes_points(image.shape[0])
        for lane_points, lane_line in zip(lanes_points, self._lane_lines):
            # Get the formato for fit_x and self._plot_y that cv2.polylines demands
            fit_points = np.array((lane_points, plot_y), np.int32).T.reshape((-1, 1, 2))
            color = (0, 0, 255) if lane_line.is_unreliable() else (255, 0, 255)
            image_with_lane_lines = cv2.polylines(image_with_lane_lines,
                                                  [fit_points],
                                                  False,
                                                  (color),
                                                  thickness=3)
        return image_with_lane_lines

    def overlay_lanes_in_perspective(self, image):
        """
        Color in the given image the area corresponding to the detected lane.
        :param image: a color image taken by the camera (shoudl be already corrected for distortion)
        :param M: the transformation matrix previously used to warp camera images to the brid-eye view;
        the method uses it inverse to project the colored polygon onto the camera perspective.
        :return: the resulting image.
        """

        assert self._M is not None
        plot_y, fit_x = self.get_lanes_points(image.shape[0])

        # If either lane is not available, give up and return the input image, unchanged
        if fit_x[0] is None or fit_x[1] is None:
            return image

        # Create an initially black image to draw the lines on
        color_warp = np.zeros_like(image).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([fit_x[0], plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_x[1], plot_y])))])
        pts = np.hstack((pts_left, pts_right))
        pts = np.squeeze(pts)
        pts = np.expand_dims(pts, 1)

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self._M, (color_warp.shape[1], color_warp.shape[0]),
                                      flags=cv2.WARP_INVERSE_MAP)
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        return result

    def overlay_thresholded(self, image):
        thresholded_color = turn_grayscale_into_color(self._thresholded)
        with_thresholded = cv2.addWeighted(image, 1, thresholded_color, 0.5, 0)
        return with_thresholded

    def overlay_additional_info(self, image, frame_n):

        '''radiuses = []
        for lane_line in self._lane_lines:
            radius = lane_line.get_curvature()
            radiuses.append(radius)'''
        radius = (self._lane_lines[0].get_curvature() + self._lane_lines[1].get_curvature()) / 2
        x1, x2 = get_lane_lines_position_at(self._lane_lines[0], self._lane_lines[1], image.shape[0] - 1)
        lane_width = abs(x1 - x2) * self._mx
        position = ((x1 + x2) / 2 - image.shape[1] // 2) * self._mx
        # top_lane_width = get_lane_width(self._lane_lines[0], self._lane_lines[1], image.shape[0] // 2) * self._mx
        to_print = '#{:d} Curvature radius={:5.0f}m, Lane width={:2.2f}m, Position={:+1.2f}m'.format(frame_n,
                                                                                                     radius,
                                                                                                     lane_width,
                                                                                                     position)
        text_color = (0, 0, 128)
        cv2.putText(image, to_print, (0, 50), self._font, 1, text_color, 2, cv2.LINE_AA)
        return image

    def process_frame(self, frame, frame_n):
        self._unprocessed = frame
        self.invalidate()
        self.get_top_view()
        if frame_n == 336:
            save_frame(self._top_view, 'output_images/top_view.png', 'Top view')
        self.get_thresholded()
        if frame_n == 336:
            save_frame(self._thresholded, 'output_images/thresholded.png', 'Thresholded')
        self.position_windows()
        self.fit_lane_lines()
        frame_with_lane = self.overlay_lanes_in_perspective(frame)
        with_thresholded = self.overlay_thresholded(frame_with_lane)
        with_windows = self.overlay_windows(with_thresholded)
        with_lane_line = self.overlay_lane_lines(with_windows)
        if frame_n == 336:
            save_frame(with_lane_line, 'output_images/interpolated.png', 'Interpolated lanes')
        with_text = self.overlay_additional_info(with_lane_line, frame_n)
        return with_text


def no_ticks(axes):
    '''
    Removes ticks and related numbers from both axis of the given matplotlib.axes.Axes
    '''
    axes.get_xaxis().set_ticks([])
    axes.get_yaxis().set_ticks([])


def save_undistorted_sample(f_name, mtx, dist, roi):
    '''
    Undirstorts the image from the given file based on camera parameters `mtx` and `dist` and saves
    the result in a .png file under `output_dir`, along with the original (distorted) image, for comparison
    '''

    img = cv2.imread(f_name)
    assert img is not None
    undistorted_img = undistort_image(img, mtx, dist, roi)

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


def save_frame(frame, f_name, title=None):
    """
    Saves the given image in a file with the given name, the image can be grayscale or BGR. 
    """

    if len(frame.shape) == 2:
        frame = turn_grayscale_into_color(frame)
    else:
        assert len(frame.shape) == 3
        frame = switch_RGB(frame)

    image_PIL = PIL.Image.fromarray(frame)
    image_PIL.save(f_name)


def parse_args():
    # Description of this program.
    desc = "Advanced Lane Detection -Project for Udacity's Self-Driving Cars Nanodegree Program."

    # Create the argument parser.
    parser = argparse.ArgumentParser(description=desc)

    # Add arguments to the parser.
    parser.add_argument("--input", required=False, default='project_video.mp4',
                        help="name of the input file with video clip to be processed")

    # Parse the command-line arguments.
    args = parser.parse_args()

    # Get the arguments.
    input_fname = args.input

    return input_fname


if __name__ == '__main__':
    input_fname = parse_args()
    output_fname = 'out_' + input_fname

    # Directory for sample output images
    output_dir = 'output_images'

    # Directory containing images for caliration
    calibration_dir = 'camera_cal'

    # Size of the checkered calibration target, columns by rows
    target_size = (9, 6)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calibrate the camera
    mtx, dist, rvecs, tvecs, roi = calibrate_camera(calibration_dir, target_size)

    # Save an image with one calibration sample along with its undistorted version
    # save_undistorted_sample(calibration_dir + '/calibration2.jpg', mtx, dist, roi)

    vidcap = cv2.VideoCapture(input_fname)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    assert fps > 0
    vertical_resolution = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    horizontal_resolution = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidwrite = cv2.VideoWriter(output_fname, fourcc=fourcc, fps=fps,
                               frameSize=(int(horizontal_resolution), int(vertical_resolution)))
    print('Source video', input_fname, 'is at', fps, 'fps with resolution of', int(horizontal_resolution), 'by',
          int(vertical_resolution),
          'pixels')

    frame_counter = 0
    # vidcap.set(cv2.CAP_PROP_POS_MSEC, 6000)
    start_time = time.time()
    processor = ImageProcessing()

    while (True):
        read, frame = vidcap.read()
        if not read:
            break
        frame_counter += 1
        if frame_counter == 336:
            save_frame(frame, 'output_images/original.png', 'Original image')
        print('Processing frame', frame_counter)
        # Un-distort the frame applying camera calibration
        undistorted_img = undistort_image(frame, mtx, dist)
        if frame_counter == 336:
            save_frame(undistorted_img, 'output_images/undistorted.png', 'Undistorted image')

        # Process the frame and find the lane lines
        processed = processor.process_frame(undistorted_img, frame_counter)
        # Write the result to the output video stream
        vidwrite.write(processed)
        if frame_counter % 100 == 0:
            pass

    print('\nProcessing rate {:.1f} fps'.format(frame_counter / (time.time() - start_time)))

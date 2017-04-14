import cv2
import matplotlib.pyplot as plt
import matplotlib as mplib
import numpy as np
import os
import glob
import math
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return new_mtx, dist, rvecs, tvecs, roi


def switch_RGB(img):
    '''
    Switches between RGB and BGR representation of a color image, and returns the result
    '''
    return img[:, :, ::-1]


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
    undistorted_img = undistort(img, mtx, dist, roi)

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


def save_undistorted(f_name, mtx, dist, roi):
    img = cv2.imread(f_name)
    assert img is not None
    undistorted_img = undistort(img, mtx, dist, roi)
    f_basename = os.path.basename(f_name)
    output_f_name = os.path.splitext(f_basename)[0] + '.png'
    cv2.imwrite(output_dir + '/undistorted-' + output_f_name, undistorted_img)
    plt.clf()  # plt.close() would generate an error 'can't invoke "event" command: application has been destroyed'


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


def undistort(img, mtx, dist, roi):
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    x, y, w, h = roi
    # cropped_img= undistorted_img[y:y + h, x:x + w]
    cropped_img = undistorted_img
    return cropped_img


def find_perspective_transform(img):
    # Begin by finding the perspective vanishing point
    lane_l = ((264, 688), (621, 431))  # Two points identifying the line going along the left lane marking
    lane_r = ((660, 431), (1059, 688))  # Two points identifying the line going along right lane marking
    v_point = find_intersect(*lane_l, *lane_r)  # Intersection of the two lines above (perspective vanishing point)

    # Determine a quadrangle in the source image
    source_h = img.shape[0]
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
    target_shape = (img.shape[0], img.shape[1])  # (rows, columns)
    target = np.float32([[source[0, 0], target_shape[0] - 1],
                         [source[0, 0], 0],
                         [source[3, 0], 0],
                         [source[3, 0], target_shape[0] - 1]])

    # Given the source and target quandrangles, calculate the perspective transform matrix
    source = np.expand_dims(source, 1)  # OpenCV requires this extra dimension
    M = cv2.getPerspectiveTransform(src=source, dst=target)
    return M


def find_gradient(gscale_img):
    """
    Returns the gradient modulus and direction absolute value for the given grayscale image.
    Modulus is scaled to be in the range of integers [0, 255], direction to be in the real numbers interval
    [0, Pi]
    """
    sobel_x = cv2.Sobel(gscale_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gscale_img, cv2.CV_64F, 0, 1, ksize=5)
    grad_size = (sobel_x ** 2 + sobel_y ** 2) ** .5
    max_grad_size = np.max(grad_size)
    grad_size = np.uint8(grad_size / max_grad_size * 255)
    grad_dir = np.abs(np.arctan2(sobel_y, sobel_x))
    return grad_size, grad_dir


def params_browser(image):
    """
    Presents a dialog that allows interactive tuning of various parameters, and shows the effect on the given image.
    The input image should be in BGR color space, already un-distorted and warped
    """

    # Go grayscale
    gscale = grayscale(image)

    # Set the grid
    grid = gridspec.GridSpec(7, 1, height_ratios=[4.5, .25, .25, .25, 2, 1, .5])

    # Plot the image
    picture = plt.subplot(grid[0, 0])
    axes_image = plt.imshow(gscale, cmap='gray')

    # Plot the sliders
    axes_modulus = plt.subplot(grid[1, 0])
    modulus_slider = Slider(axes_modulus, 'Grad modulus', valmin=0, valmax=255, valinit=128)
    axes_direction_min = plt.subplot(grid[2, 0])
    min_direction_slider = Slider(axes_direction_min, 'Min', valmin=0, valmax=1, valinit=0,
                                  valfmt='%1.3f')  # TODO fix slidermax
    axes_direction_max = plt.subplot(grid[3, 0])
    max_direction_slider = Slider(axes_direction_max, 'Max', valmin=0, valmax=1, valinit=1, valfmt='%1.3f',
                                  slidermin=min_direction_slider)

    # Plot the radio buttons
    axes_color_space = plt.subplot(grid[4, 0])
    radio_cspace = RadioButtons(axes_color_space, ('RGB', 'YUV', 'HSV', 'HLS', 'Lab'), active=1)
    axes_channel = plt.subplot(grid[5, 0])
    radio_channel = RadioButtons(axes_channel, ('1', '2', '3'), active=0)
    axes_enable_grad = plt.subplot(grid[6, 0])
    radio_grad = RadioButtons(axes_enable_grad, ('Gradient', 'Channel'), active=0)

    plt.tight_layout(h_pad=0)
    plt.subplots_adjust(left=.2, right=.9)

    def update(_):
        # Start from the undistorted color image `warped` and convert it of color space if necessary
        label = radio_cspace.value_selected
        conversion = {'RGB': None,
                      'YUV': cv2.COLOR_BGR2YUV,
                      'HSV': cv2.COLOR_BGR2HSV,
                      'HLS': cv2.COLOR_BGR2HLS,
                      'Lab': cv2.COLOR_BGR2LAB}
        converted = cv2.cvtColor(image, conversion[label]) if conversion[label] is not None else np.copy(image)

        take_gradient = True if radio_grad.value_selected == 'Gradient' else False

        # Convert it to grayscale by keeping the required channel and discarding the other two
        channel = int(radio_channel.value_selected) - 1
        # RGB images are in memory as BGR, fix the channel number accordingly
        if label == 'RGB':
            channel = 2 - channel
        gscale = converted[:, :, channel]

        min_direction = min_direction_slider.val
        max_direction = max_direction_slider.val
        thresholded = np.zeros_like(gscale)
        if take_gradient:
            min_direction *= math.pi / 2
            max_direction *= math.pi / 2
            modulus = modulus_slider.val
            # Determine the gradient
            grad_size, grad_dir = find_gradient(gscale)
            # Threshold the gradient
            thresholded[(grad_size >= modulus) & (((grad_dir >= min_direction) & (grad_dir <= max_direction)) | (
                (grad_dir >= math.pi - max_direction) & (grad_dir <= math.pi - min_direction)))] = 255
        else:
            min_direction = round(min_direction * 255)
            max_direction = round(max_direction * 255)
            thresholded[(gscale >= min_direction) & (gscale <= max_direction)] = 255

        # Display the udpated image, after thresholding
        axes_image.set_data(thresholded)
        plt.draw()

    # Register call-backs with widgets
    modulus_slider.on_changed(update)
    min_direction_slider.on_changed(update)
    max_direction_slider.on_changed(update)
    radio_cspace.on_clicked(update)
    radio_channel.on_clicked(update)
    radio_grad.on_clicked(update)

    plt.show()


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(thresholded, window_width, window_height, margin):
    min_acceptable = 100
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(thresholded[int(3 * thresholded.shape[0] / 4):, :int(thresholded.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    last_good_l_center = l_center
    r_sum = np.sum(thresholded[int(3 * thresholded.shape[0] / 4):, int(thresholded.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(thresholded.shape[1] / 2)
    last_good_r_center = r_center

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(thresholded.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            thresholded[
            int(thresholded.shape[0] - (level + 1) * window_height):int(thresholded.shape[0] - level * window_height),
            :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(last_good_l_center + offset - margin, 0))
        l_max_index = int(min(last_good_l_center + offset + margin, thresholded.shape[1]))
        goodness = np.max(conv_signal[l_min_index:l_max_index])
        if goodness >= min_acceptable:
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            last_good_l_center = l_center
        else:
            l_center = None
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(last_good_r_center + offset - margin, 0))
        r_max_index = int(min(last_good_r_center + offset + margin, thresholded.shape[1]))
        goodness = np.max(conv_signal[r_min_index:r_max_index])
        if goodness >= min_acceptable:
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            last_good_r_center = r_center
        else:
            r_center = None
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calibrate the camera
    mtx, dist, rvecs, tvecs, roi = calibrate_camera()

    # Save an image with one calibration sample along with its undistorted version
    save_undistorted_sample(calibration_dir + '/calibration2.jpg', mtx, dist, roi)

    #############################################################
    # test_image = 'test_images/straight_lines2.jpg'
    test_image = 'test_images/test1.jpg'
    save_undistorted(test_image, mtx, dist, roi)

    # Load a test image (for now)
    img = cv2.imread(test_image)
    assert img is not None

    # Un-distort it applying camera calibration
    undistorted_img = undistort(img, mtx, dist, roi)

    # Determine the pespective transform transform
    M = find_perspective_transform(undistorted_img)

    # Apply it to the source image
    warped = cv2.warpPerspective(undistorted_img, M, undistorted_img.shape[1::-1])

    # params_browser(warped)

    hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    grad_size, grad_dir = find_gradient(l_channel)
    min_grad_size = 18
    min_grad_dir = 0
    max_grad_dir = 0.367 * math.pi / 2
    thresholded_grad = np.zeros_like(l_channel)
    thresholded_grad[(grad_size >= min_grad_size) & (((grad_dir >= min_grad_dir) & (grad_dir <= max_grad_dir)) | (
        (grad_dir >= math.pi - max_grad_dir) & (grad_dir <= math.pi - min_grad_dir)))] = 255
    r_channel = warped[:, :, 2]
    min_r = int(0.881 * 255)
    max_r = 255
    thresholded_r = np.zeros_like(r_channel)
    thresholded_r[(r_channel >= min_r) & (r_channel <= max_r)] = 255
    thresholded = np.zeros_like(l_channel)
    thresholded[(thresholded_grad == 255) | (thresholded_r == 255)] = 255
    # plt.imshow(thresholded, cmap='gray')
    # plt.show()

    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching
    window_centroids = find_window_centroids(thresholded, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        # l_points = np.zeros_like(warped)
        # r_points = np.zeros_like(warped)
        l_points = np.zeros_like(thresholded)
        r_points = np.zeros_like(thresholded)

        l_lane_points = np.zeros_like(thresholded)
        r_lane_points = np.zeros_like(thresholded)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            # Add graphic points from window mask here to total pixels found
            if window_centroids[level][0] is not None:
                l_mask = window_mask(window_width, window_height, thresholded, window_centroids[level][0], level)
                l_lane_points[(l_mask==1) & (thresholded==255)] = 255
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
            if window_centroids[level][1] is not None:
                r_mask = window_mask(window_width, window_height, thresholded, window_centroids[level][1], level)
                r_lane_points[(r_mask==1) & (thresholded==255)] = 255
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((thresholded, thresholded, thresholded)),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((thresholded, thresholded, thresholded)), np.uint8)

    l_point_coords = np.where(l_lane_points==255)
    r_point_coords = np.where(r_lane_points == 255)
    '''assert len(l_point_coords[0])==len(l_point_coords[1])
    for i in range(len(l_point_coords[0])):
        assert l_lane_points[l_point_coords[0][i], l_point_coords[1][i]] == 255'''

    l_fit = np.polyfit(l_point_coords[0], l_point_coords[1], 2)
    r_fit = np.polyfit(r_point_coords[0], r_point_coords[1], 2)
    ploty = np.linspace(0, thresholded.shape[0] - 1, thresholded.shape[0])
    l_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
    r_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]
    plt.plot(l_fitx, ploty, color='yellow')
    plt.plot(r_fitx, ploty, color='yellow')



    # Display the final results
    # plt.imshow(output)
    plt.imshow(r_lane_points+l_lane_points, cmap='gray')
    # plt.title('window fitting results')
    plt.show()


if __name__ == '__main__':
    main()

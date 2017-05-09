# README

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistored-calibration2.png "Undistorted calibration image"
[image2]: ./output_images/undistorted.png "Undistorted frame"
[image3]: ./output_images/thresholded.png "Thresholded"
[image4]: ./output_images/top_view.png "Bird-eye view"
[image5]: ./output_images/interpolated.png "Interpolated lane lines"
[image7]: ./output_images/thresholded_rec.png "Finding lane line starting point"
[image8]: ./output_images/filter.png "Gaussian filter"
[image9]: ./output_images/lane.png "Projected lane"
[image10]: ./output_images/unwarped-1.png "Vanishing point"
[image11]: ./output_images/unwarped-2.png "Source points"

## To Run the Program

`python3 main.py --input <file_name>` 

If no input file name is provided, it defaults to `project_video.mp4`.

The output is found in the same directory as the input file, with name appended with `_out`; e.g. `project_video_out.mp4`.

The program has been tested under **Ubuntu 16.04** with **Python 3.5** and **OpenCV 3.2**.

## Project Content

 - `challenge_video.mp4` input video clip provided by Udacity.
 - `harder_ challenge_video.mp4` input video clip provided by Udacity.
 - `main.py` the project program for lane detection.
 - `out_challenge_video.mp4` sample of program output.  
 - `out_project_video.mp4` sample of program output.
 - `project_video.mp4` input video clip provided by Udacity.
 - `README.md` this file and write-up for the project.
 - `./camera_cal` images for camera calibration.
 - `./output_images` images for this write-up, see list below.
 - `./test_images` images useful for testing and parameter tuning provided by Udacity.
 
The content of `./output_images`:
 - `filter.png` chart with the Gaussian filter used for convolution, to detect lane line markers in the thresholded image.
 - `interpolated.png` debugging output of the program, it shows the thresholded lane lines, sliding windows and interpolated lane lines from bird-eye view, overlaid onto the camera image.
 - `lane.png` sample frame of the program video output, with the lane highlighted.
 - `original.png` input frame from the camera used to produce `lane.png`.
 - `thresholded.png` sample of result of thresholding the bird-eye view.
 - `thresholded_rec.png` it highlights the portion of the thresholded image used to determine the x coordinate of the left lane line at the bottom of the image, through convolution.
 - `top_view.png` sample of the result of transforming the undistorted camera image to a bird-eye perspective.
 - `undistorted_calibration2.png` eample of image used for camera calibration, before and after it was undistorted.
 - `undistorted.png` result of applying camera calibration to correct `original.png`.
 - `unwarped-1.png` illustration of the determination of the perspective vanishing point, starting from an undistorted camera image.
 - `unwarped-1.pptx` the same as a slide.
 - `unwarped-2.png` illustration of the determination of source points for transformation to bird-eye perspective.
 - `unwarped-2.pptx` the same as a slide.
 


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the requirements individually and describe how I addressed them in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

This is it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Function `calibrate_camera()` reads JPEG files from a given directory and uses them to calculate the camera calibration parameters.

It converts every image to grayscale and then finds the corners of a checkered calibration pattern with `cv2.findChessboardCorners()`. For every input image, it compiles the list of found corner coordinates (image points), and the coordinates those points should have after distortion correction (object points). The function then calls `cv2.calibrateCamera()`, which compares the found image points coordinates with the object points coordinates, and outputs the camera matrix and distortion coefficients.
 
 Before returning the required parameters and camera matrix, function `calibrate_camera()` evaluates how good they are: it uses them to transform the object points into image points, with `cv2.projectPoints()`, and then it measures the average distance between the transformed points and the respective actual image points (as detected in the calibration images). The closer to 0 the average distance, the better the calibration quality. This measure could be used to calibrate the camera with different and/or additional images, to see if calibration results improve.
 
 The image below shows one of the calibration images, and its undistorted version after camera calibration. The image has been produced with function `save_undistorted_sample()`.
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
After camera calibration, a loop in `main.py` processes video frames one at a time. It undistorts the frame calling `undistort_image()` and then passes the undistorted frame to method `ImageProcessing.process_frame()`. The image below is a camera frame as returned by `undistort_image()`.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Most of the image elaboration is done in two classes, `ImageProcessing`, that drives every step of the processing and stores the resulting images, and LaneLine, which finds lane line markers by interpolation and stores the results.
 
Method `ImageProcessing.get_top_view()` transforms the image perspective, going from the camera viewpoint to a bird-eye view. In such a way, lines that are parallel on the road and converge to a point in the camera perspective, are parallel again in the transformed image. Previous camera calibration and image undistortion are instrumental to get lines that are parallel on the road actually parallel in the bird-eye view.

For perspective transformation I used OpenCV functions `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`: the former computes a transformation matrix, and the latter applies it to the (undistorted) image.

In order to compute the transformation matrix, OpenCV requires coordinates of four points on the image, and corresponding coordinates after transformation.

This is how the four points on the image are chosen. Starting from an undistorted camera image with straight lane lines, and with the help of a graphics editor, I have chosen a pair of points on each lane line, that after perspective transformation should be the vertices of a rectangle; see image below, where the for points are named A, B, C and D.

![alt text][image10]

The program determines the intersection of the two lines going through AD and BC respectively, obtaining the vanishing point of the camera perspective, V in the image above. The program then computes four points to be used for perspective transformation, A' and D' lying along the AV line, and B' and C' lying along the BV line, such that, after transformation, they should delimit a rectangle in the bird-eye view. See in image below.

![alt text][image11]

That way, once I set the four points A, B, C and D, I could easily tune the distance of segment D'C' from the top of the image, and of segment A'B' from the bottom. At the bottom, I wanted to leave the car hood out of the transformed image; at the top, I wanted to include as much as possible of the road, and the lane lines, before they become too blurry to be useful for further processing. 

Table below lists coordinates of the four points, before and after perspective transformation.    

| Point| Source        | Target   | 
|:---:|:-------------:|:-------------:| 
|A'| 290.4, 669      | 290.4, 719        | 
|B'| 1029.5, 669      | 1029.5, 719        |
|C'| 700.4, 457     | 1029.5, 0      |
|D'| 584.9, 457      | 290.4, 0      |

Below you can see the result of transformation to bird-eye view of an undistorted image. Pixels closer to the top are at a greater distance from the car, and are blurrier after the perspective transformation.

![alt text][image4]

I ensured the image maintains its original resolution after transformation (1280x720 pixels), which allows me to overlay the resulting image on the camera image, very useful for subsequent parameters tuning and debugging.

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Method `ImageProcessing.get_thresholded()` starts from a bird-eye view image (undistorted and perspective transformed), and thresholds it producing a black-and-white image with the same resolution. Image below is an example of the output.

![alt text][image3]

The method converts the image into a HSV color space, and then thresholds its individual channels and the x gradient of the V channel. The operation is repeated with different thresholding intervals to match white and yellow lines, and to be more likely to match them if they are aligned close to vertical. If a pixel passes thresholding with at least one interval, then it is included in the thresholded output.

Method `find_x_gradient()` computes and returns the absolute value of the gradient of a grayscale image in the x direction using `cv2.Sobel()` and a kernel of size 5; result is scaled to range between 0 and 255 in every given image.  

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

Method `ImageProcessing.position_windows()` identifies areas in the thresholded image that are likely to contain pixels from the lane lines. Then method `ImageProcessing.fit()` interpolates those pixels with two parabolas, one per lane line.

In the image below the undistorted image is overlaid with the output of thresholding, and of the two above methods. This kind of image has been invaluable for parameters tuning and debugging. Parabolas interpolating the lane lines, as seen in the bird-eye perspective, are in fuchsia. Rectangles delimit areas of the image (sliding windows) believed to contain pixels belonging to lane lines. The number next to each sliding window is an estimate of its goodness, the higher the number, the more lane line points are probably in it. Sliding windows with a goodness under a certain threshold are ignored for further processing, and drawn in red in the image below.  

![alt text][image5]

Method `ImageProcessing.position_windows()` looks for thresholded pixels likely to be part of either lane line starting from the bottom of the image, and moving upward. It partitions the image in 9 horizontal bands of the same height, and in every band it slides a window in the surrounding of a given x coordinate (let's call it x0), looking for lane line pixels. By limiting the search to those surroundings, it performs faster than looking in the whole band.

In the first image frame, to find x0 for the lowest band and the left lane line, it convolves a filter with Gaussian shape with the lowest left eighth of the image (highlighted in blue in the picture below), and sets x0 to the value of x that maximises the convolution result.

![alt text][image7]

It does the same on the lowest right eighth of the image to find x0 for the right lane line at the bottom band. Implementation of this is in method `LanelLine.recenter()`. Chart below shows values of the adopted Gaussian filter, which has size 100x1 pixels.

![alt text][image8]

Method `ImageProcessing.position_windows()` then starts looking in every band starting from the bottom; it slides a rectangular window in a given surrounding of x0 finding the window position that maximises convolution with the Gaussian filter. Thresholded pixels that are inside the window are believed to be part of the lane line. However, if the result of convolution is below a set threshold, the window is marked as a bad match, and pixels inside it are ignored instead (rectangles in red in the image above).

After finding the optimal sliding window positions for the left and right lane lines, it proceeds with the next band higher up. Now as x0 for that band and lane line it uses the centre of the highest window in a band below which was not a bad match.

If the windows at the bottom band were good match, the respective x0 will be kept to process the bottom band of the next frame; if either of them was a bad match, then processing of the next frame will re-calculate its x0 like for the first frame, i.e. convolving the whole lowest left or right eighth of the image.

Once determination of sliding windows is complete for the frame in every band, method `ImageProcessing.fit()` interpolates thresholded pixels that are in any window, for the left and right lane line respectively. Interpolation is done by Least Squares Mean with a parabola. I have experimented with RANSAC, but while it would correctly ignore many outliers, it also had a tendency to ignore pixels that are part of the lane line, when the lane line is bent.

After interpolation, method `ImageProcessing.fit()` calls `LaneLine.check_reliability()` and `LaneLine.check_reliability_against()` to evaluate the goodness of the determined lane lines. Those two methods apply a number of criteria such as how many sliding windows were marked as bad match, the lane width, respective radius of curvature of the two lane lines. As a result, interpolated lane lines that don't pass the criteria are marked as unreliable. An unreliable lane line in a given frame is replaced by the same lane line in the previous frame.

In order to smooth drawing of the result from frame to frame, method `LaneLine.fit()` weight averages the interpolated parabola with the one from previous frame. More specifically, let `coefficients` be a Numpy array with the interpolated parabola coefficients, like `[a, b, c]` where `x=a*(y**2)+b*y+c`, and let `prev_coefficients` be the coefficients of the corresponding parabola from the previous frame, then `coefficients` is updated as:
 
 `coefficients = (1-smoothing) * coefficients + smoothing * prev_coefficients`
 
 and `prev_coefficients` as:
 
 `prev_coefficients = coefficients`
 
 where `smoothing` is a set floating point number in the interval `[0, 1)`; when it is closer to 1, smoothing is more effective, but interpolated lane lines may take more frames to catch up with actual lane lines, that are moving and bending in the image. In the current implementation it is set to 0.5. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Function `measure_curvature(coefficients, y0, mx, my)` calculates and returns the curvature radius, in meters, for the parabola with given `coefficients`, as measured at the y coordinate `y0`.

Coefficients and `y0` must assume the reference coordinate system to be in pixels, and the function performs the necessary conversions from pixels to meters, using the given conversion factors `mx` and `my`. The latter are given in meters/pixel, and state, for one pixel in the bird-eye view image, to how many meters it corresponds on the road, in the x and y direction respectively. 

I estimated `mx` and `my` based on images taken from the camera, and assumptions on the lane width and lane line markings length, as expected on California highways. I have set them to 3.66/748 and 3.48/93 meters/pixel respectively.

The formula to compute the curvature radius, as implemented, is:

```
a = mx / (my ** 2) * coefficients[0]
b = mx / my * coefficients[1]
Y0 = y0 * my

radius = ((1 + (2 * a * Y0 + b) ** 2) ** 1.5) / (2 * a)
```

It first converts the parabola reference system from pixels to meters, and then computes the radius. As one may expect, it doesn't depend on the third parabola coefficient `coefficients[2]`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Method `ImageProcessing.overlay_lanes_in_perspective()` paints an overlay of the lane on the (undistorted) camera image, and returns the result. It draws a polygon with `cv2.fillPoly()` on a bird-eye view, and then transforms it to the camera perspective using the inverse of the transformation matrix previously used to go to the bird-eye view.

Image below is an example of the frame as produced for the output video. On top it reports a progressive frame counter, the lane curvature radius (a negative number indicates that the center of curvature is to the left), the lane width and the distance of the car from the center of the lane (a negative number indicates the car is to the left of the lane center).


![alt text][image9]

Lane curvature radius is computed as the average between the curvatur radii of the two lane lines.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

A video clip with the program output can be found [here](./out_project_video.mp4) and, for a more "difficult" input, [here](./out_challenge_video.mp4). They may be produced again and overwritten by running the program.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Source code profiling indicates that thresholding is the "long" operation in the image processing pipeline. In the current implementation, it is done on the whole frame. A possible improvement is to threshold only those parts of the image that might fall into one of the sliding windows.

While the algorithmic approach may seem quite general, its main weakness is the needed tuning of thresholding parameters, used to detect lane lines of different colors and orientation, in different light conditions and on different pavement color. Tuning allows the program to perform well on the adopted input, but may be brittle when confronted with input taken on different roads, under different light and atmospheric conditions.

A more robust approach may try to detect lane lines or segments of lane lines before interpolation, e.g. through a Hough transform. Interpolation would take as input the Hough transform output, possibly using splines instead of parabolas, or could be skipped altogether.

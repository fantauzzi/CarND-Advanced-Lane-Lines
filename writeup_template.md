##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Function `calibrate_camera()` reads JPEG files from a given directory and use them to calculate the camera calibration parameters.

It converts every image to grayscale and then finds the corners of a checkered calibration pattern with `cv2.findChessboardCorners()`. For every input image, it compiles the list of found corner coordinates (image points), and the coordinates those points should have after distortion correction (object points). It then calls `cv2.calibrateCamera()`, which compares the found image points coordinates with the object points coordinates, and outputs the camera matrix and distortion coefficients. After that, it calls `cv2.getOptimalNewCameraMatrix()` to ensure every pixel from the original image is still in the undistorted image (at the expense of possible black pixels at the edges of the undistorted image).
 
 Before returning the required parameters and camera matrix, function `calibrate_camera()` evaluates how good they are: it uses them to transform the object points into image points, with `cv2.projectPoints()`, and then it measures the average distance between the transformed points and the respective actual image points (as detected in the calibration images). The closer to 0 the average distance, the better the calibration quality. This measure could be used to calibrate the camera with different and/or additional images, to see if the calibration result improves.
 
 The image below shows one of the calibration images, and its undistorted version after camera calibration. The image has been produced with function `save_undistorted_sample()`

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
After camera calibration, a loop in `main.py` processes video frames one at a time. It undistorts the frame calling `undistort_image()` and then passes the undistorted frame to method `ImageProcessing.process_frame()`. The image below is a frame as returned by `undistort_image()`.

![alt text][image2]

####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Most of the image elaboration is done in two classes, `ImageProcessing`, that drives every step of the processing and stores the resulting images, and LaneLine, which finds lane line markers by interpolation and stores the results.
 
Method `ImageProcessing.get_top_view()` transforms the image perspective, going from the camera viewpoint to a bird-eye view. In such a way, lines that are parallel on the road and converge to a point in the camera perspective, are parallel again in the transformed image. Previous camera calibration and image undistortion are instrumental to get lines that are parallel on the road actually parallel in the bird-eye view.

For perspective transform I used OpenCV functions `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`: the former computes a transformation matrix, and the latter applies it to the (undistorted) image.

In order to compute the transformation matrix, OpenCV requires coordinates of four points on the image, and corresponding coordinates after transformation.

This is how the four points on the image are chosen. Starting from an undistorted camera image with straight lane lines, and with the help of a graphics editor, I have chosen a pair of points on either lane, that after perspective transformation should be the vertices of a rectangle; see image below, where the for points are named A, B, C and D.

The program determines the intersection of the two lines going through the two pairs of points respectively (AD and BC), obtaining the vanishing point of the camera perspective, V in the image above. The program then computes four points to be used for the perspective transformation, A' and D' laying along the AV line, and B' and C' laying along the BV line, such that after transformation they should delimit a rectangle.

That way, once I set the four points A, B, C and D, I could easily tune the distance of segment D'C' from the top of the image, and of segment A'B' from the bottom. At the bottom, I wanted to leave the car hood out of the transformed image; at the top, I wanted to include as much as possible of the road, and the lane lines, before they become too blurry to be useful for further processing. 

Table below lists coordinates of the four points, before and after perspective transformation.    

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

Image below shows the transformation result on a camera frame.

![alt text][image4]

I ensured the image maintains the original resolution after transformation (1280x720 pixels), which allows me to overlay the resulting image on the camera image, very useful for subsequent parameters tuning and debugging.

####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Method `ImageProcessing.get_thresholded()` starts from a bird-eye view image (undistorted and perspective transformed), and thresholds it producing a black-and-white image with the same resolution. Image below is an example of the output.

![alt text][image3]

The method converts the image into a HSV color space, and then thresholds its individual channels and the x gradient of the V channel. The operation is repeated with different thresholding intervals to match white and yellow lines, and to be more likely to match them if they are aligned close to vertical. If a pixel passes thresholding with at least one interval, then it is included in the thresholded output. 

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Method `ImageProcessing.position_windows()` indentifies areas in the thresholded image that are more likely to contain pixels from the lane lines. Method `ImageProcessing.fit()` interpolates those pixels with two parabolas, one per lane line. 

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


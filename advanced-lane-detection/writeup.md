## Advanced lane detection for self driving cars

### This project aims to create a pipeline with Computer Vision techniques to detect lanes on the road.

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

[image1]: ./output_images/chessboard_undistorted.png "Chessboard undistortion"
[image2]: ./output_images/undistorted.png "Undistorted"
[image3]: ./output_images/gradients_and_color.png "Gradients and color processing"
[image4]: ./output_images/combined.png "Gradients and color processing combined"
[image5]: ./output_images/warped.png "Bird's eye view transformation"
[image6]: ./output_images/sliding_windows.png "Lane finding with sliding windows"
[image7]: ./output_images/previous_poly.png "Lane finding with previous polynomial coefficients"
[image8]: ./output_images/result.png "Unwarped detections on original image"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

The code for the whole pipeline is in [advanced_lane_detection.ipynb](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/advanced_lane_detection.ipynb) and [helpers.py](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/helpers.py)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The first step is to calibrate the camera. This process is aided by some images of a black and white chessboard from different perspectives. Using OpenCV's `findChessboardCorners` function, we are able to compute the corners, i.e. image points, to calibrate the camera. These corners refer to the inner corners of the chessboard. Therefore, we should create the object points as an indexes array that contains width*height (of the chessboard) elements. By assuming the chessboard is not moving in the plane z, these image points together with the coordinates in object points will be enough to calibrate the camera using OpenCV's `calibrateCamera` function and extract the camera matrix (mtx) and the distorsion coefficients (dist) parameters.

![alt text][image1]

See *Camera calibration* section of [advanced_lane_detection.ipynb](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/advanced_lane_detection.ipynb) and lines 10 through 78 of [helpers.py](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/helpers.py)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the camera matrix (mtx) and the distorsion coefficients (dist) parameters we can undistort any image taken with this camera.
![alt text][image2]

See *Create binary image* section of [advanced_lane_detection.ipynb](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/advanced_lane_detection.ipynb) and lines 81 through 110 of [helpers.py](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/helpers.py)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used a combination of sobel x and y thresholds, sobel magnitude threshold, sobel direction threshold and saturation threshold (for yellow lanes). Here's an example of my output for this step.

![alt text][image3]

The result of combining all these features can be seen in the following image.

![alt text][image4]

See *Create binary image* section of [advanced_lane_detection.ipynb](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/advanced_lane_detection.ipynb) and lines 113 through 210 of [helpers.py](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/helpers.py)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To perform a perspective transformation on the image, we have to define four points in a given image that will form a rectangle in the destination image. In this step we have to kepp in mind that we are trying to get a new image from a bird's eye view perspective. Providing these points we can get the transformation matrix M with OpenCV's `getPerspectiveTransform` to later transform the image with `warpPerspective` function.

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

See *Perspective transform* section of [advanced_lane_detection.ipynb](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/advanced_lane_detection.ipynb) and lines 213 through 250 of [helpers.py](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/helpers.py)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial

Two method were implemented to detect the lanes in the warped image. First, the sliding windows method will find the lanes using the horizontal histogram of the bottom of the image. I accounted for very tight curves since the windows may hit the sides of the image before arriving to the top. Then I try to fit a second order polynomial and return the polynomial coefficients.

![alt text][image6]

Once the lanes have been found, there is not need to run the sliding windows on each frame. On the other hand, we can look for new lane pixels within a margin of the previous lanes. 

![alt text][image7]

See *Sliding windows* and *Margin within previous lanes* section of [advanced_lane_detection.ipynb](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/advanced_lane_detection.ipynb) and lines 253 through 517 of [helpers.py](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/helpers.py)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Using the curvature of lane equation given the polynomial coefficients I calculate both the radius for the left lane and for the right lane. Then I average them and return the result.

See *Radius of curve* section of [advanced_lane_detection.ipynb](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/advanced_lane_detection.ipynb) and lines 520 through 537 of [helpers.py](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/helpers.py)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

See *Result* section of [advanced_lane_detection.ipynb](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/advanced_lane_detection.ipynb)

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link](https://youtu.be/sqobICfr758) (YouTube) to the project video.

See *Video pipeline* section of [advanced_lane_detection.ipynb](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/advanced_lane_detection.ipynb) and lines 540 through 810 of [helpers.py](https://github.com/josh31416/self-driving-car-nanodegree-at-udacity/blob/master/advanced-lane-detection/helpers.py)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had some problems of noise in the binary features step, specially in the saturacion step. I could solve this problem by setting the saturation to 0 where the brightness was not high enough.

Additionally, I implemented a function to average the last coefficients to avoid jitter, specially when dealing with discontinuous lanes.

Also, I thought there would be a problem in the sliding windows function if the curvature of a lane was too large. In this case, the windows would overflow out of the image. I simply limited the window movement to the size of the image.

The pipeline fails in case the are different asphalt colors on the road and with shadows that make the lanes too dark, like bridge shadows. It also fails were the lanes become covered by leaves, for example.

More hyperparameter tuning could be applied or some image enhancing to tackle the shadows case. However, the solution cannot be run in real time with limited computing power. The pipeline was running at 5 frames per second on my laptop. This is a real problem.

I believe this technique can be used to help label some videos and train a deep learning model. One that will hopefully have better results and one that can be run in real time (>24-30 frames per second)

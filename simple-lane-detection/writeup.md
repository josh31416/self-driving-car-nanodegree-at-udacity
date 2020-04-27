# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

Firstly, I extract do some image processing taking into account white and yellow lines. White lines are easily detectable in the grayscale image. However, some saturation filtering is needed to extract the yellow lanes. That is the reason why I transform the image to HSL colorspace. To help in the yellow detection, I turn up the gamma of the yellow image a little bit.

Secondly, I blur the image with a 7x7 kernel and extract the edges with Canny edge detection. The upper threshold is really high so that there is as little noise as possible.

After that, I mask the image to crop the ROI (region of interest) and detect the lanes with Hough Lines. It is worth mentioning the high value of max_line_gap (200). This is to allow the lines to form in the discontinuous case of lanes.

Finally, the draw_lines method is modified to compute the average lane for each side, both left and right. Lines that have a slope below a threshold are discarded since they are too horizontal to be lanes. Once the average of the slope and points is computed, I calculate the points both for the bottom of the ROI and the top.


### 2. Identify potential shortcomings with your current pipeline


The pipeline doesn't account for signs on the road or any other color of lanes. That could be a problem.

Another shortcoming is that it doesn't work with curved lanes. This is specially noticeable in the optional challenge video. The straight lanes tremble tryining to adjust to the curved lane but they can't.


### 3. Suggest possible improvements to your pipeline

- To account for curved lanes.
- To compute the average of detected lanes also in the past to eliminate infrequent incorrect detections.

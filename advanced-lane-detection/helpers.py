import numpy as np
import cv2
import math
import os
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


def calibrate_camera(object_points, image_points, img_size, pickle_save_path=None):
    """
    Calibrates camera using OpenCV's calibrate camera function

    @param object_points: array of object points needed to calibrate the camera
    @param image_points: array of image points needed to calibrate the camera
    @param img_size: image shape in the form of (width, height) to initialize camera matrix
    @param pickle_save_path (optional): path to save a pickle file containing 
        the camera calibration result

    @returns camera calibration result (ret, mtx, dist, rvecs, tvecs)
    """

    retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)
    # Save to pickle file if pickle_save_path is defined
    if pickle_save_path is not None:
        params = {}
        params["mtx"] = mtx
        params["dist"] = dist
        params["rvecs"] = rvecs
        params["tvecs"] = tvecs
        pickle.dump(params, open(pickle_save_path, "wb"))

    return mtx, dist, rvecs, tvecs


def find_chessboard_corners(chessboard_height, chessboard_width, images_regex_path, save_path=None):
    """
    Returns parameters needed to calibrate a camera using OpenCV's findChessboardCorners function.

    You should provide the algorithm with at least 20 pictures of a chessboard image 
    taken from different angles.

    @param chessboard_height: Number of vertical inner corners of the chessboard image
    @param chessboard_width: Number of horizontal inner corners of the chessboard image
    @param images_regex_path: Path to dir containing the calibration images with regex for `glob`
    @param save_path (optional): Path to save calibration images with chessboard corners drawn

    @returns objectPointsArray, imagePointsArray: arrays containing object points and image points
        needed for calibration. Use with OpenCV `calibrateCamera` function
    """

    # Prepare object points
    object_points = np.zeros((chessboard_width*chessboard_height, 3), np.float32)
    object_points[:,:2] = np.mgrid[0:chessboard_width, 0:chessboard_height].T.reshape(-1, 2)

    # Arrays for storing object points and image points from all the images
    object_points_array, image_points_array = [], []

    images = glob.glob(images_regex_path)

    for idx, filename in enumerate(images):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        retval, corners = cv2.findChessboardCorners(gray, (chessboard_width, chessboard_height), None)
        # If the corners were found, add object and image points
        if retval:
            object_points_array.append(object_points)
            image_points_array.append(corners)

            if save_path is not None:
                cv2.drawChessboardCorners(img, (chessboard_width, chessboard_height), corners, retval)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, 'chessboard_corners'+str(idx)+'.jpg'), img)

    return object_points_array, image_points_array


def undistort_image(img, calibration_params=None, path_to_pickle_file=None):
    """
    Undistorts an image given the camera calibration parameters

    @param img: OpenCV BGR image
    @param calibration_params (optional): dictionary containing calibration parameters.
        It should contain at least `mtx` and `dist` keys. (if not defined path_to_pickle_file
        should be defined)
    @param path_to_pickle_file (optional): path to pickle file that holds a dictionary 
        containing calibration parameters. It should contain at least `mtx` and `dist` keys.
        (if not defined calibration_params should be defined)

    @returns image with distortion correction
    """

    # check that either calibration_params or path_to_pickle_file are defined
    assert calibration_params is not None or path_to_pickle_file is not None

    mtx, dist = None, None
    if path_to_pickle_file is not None:
        params = pickle.load(open(path_to_pickle_file, "rb")) 
        mtx = params["mtx"]
        dist = params["dist"]
    else:
        mtx = calibration_params["mtx"]
        dist = calibration_params["dist"]

    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist


def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Computes a binary mask with the absolute value of a Sobel operator

    @param img: OpenCV BGR image
    @param orient ('x'|'y'): orientation of Sobel operator
    @param sobel_kernel: Sobel operator kernel size (must be odd)
    @param thresh: tuple with minimum and maximum thresholds, range [0-255]

    @returns binary mask with pixel activations
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, int(orient=='x'), int(orient=='y'), ksize=sobel_kernel)
    sobel_abs = np.absolute(sobel)
    scaled_sobel_abs = np.uint8(255 * sobel_abs/np.max(sobel_abs))
    mask = np.zeros_like(scaled_sobel_abs)
    mask[(scaled_sobel_abs > thresh[0]) & (scaled_sobel_abs < thresh[1])] = 1

    return mask


def mag_sobel_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    """
    Computes a binary mask with the magnitude of the gradient

    @param img: OpenCV BGR image
    @param sobel_kernel: Sobel operator kernel size (must be odd)
    @param thresh: tuple with minimum and maximum thresholds, range [0-255]

    @returns binary mask with pixel activations
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_abs = (sobelx**2 + sobely**2)**.5
    scaled_sobel_abs = np.uint8(255 * sobel_abs/np.max(sobel_abs))
    mask = np.zeros_like(scaled_sobel_abs)
    mask[(scaled_sobel_abs > thresh[0]) & (scaled_sobel_abs < thresh[1])] = 1

    return mask


def dir_sobel_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Computes a binary mask with the direction of the gradient

    @param img: OpenCV BGR image
    @param sobel_kernel: Sobel operator kernel size (must be odd)
    @param thresh: tuple with minimum and maximum thresholds, range [0-pi/2]

    @returns binary mask with pixel activations
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    abs_grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    mask = np.zeros_like(abs_grad_dir)
    mask[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1

    return mask


def saturation_threshold(s_channel, thresh=(0, 255)):
    """
    Computes a binary mask for some saturation thresholds

    @param s_channel: saturation channel of a HLS color space image
    @param thresh: tuple with minimum and maximum thresholds, range [0-pi/2]

    @returns binary mask with pixel activations
    """

    mask = np.zeros_like(s_channel)
    mask[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    return mask


def hue_threshold(h_channel, thresh=(15, 35)):
    """
    Computes a binary mask for some color thresholds

    @param h_channel: hue channel of a HLS color space image
    @param thresh: tuple with minimum and maximum thresholds, range [0-pi/2]

    @returns binary mask with pixel activations
    """

    mask = np.zeros_like(h_channel)
    mask[(h_channel >= thresh[0]) & (h_channel <= thresh[1])] = 1

    return mask


def get_perspective_transform(img, src, dst):
    """
    Calculates the perspective transformation matrix

    @param img: OpenCV BGR image
    @param src: numpy array of 4 points that will represent a rectangle
        in the warped image
    @param dst: numpy array of 4 points that form the rectangle to which
        the src points will be mapped

    @returns perspective transformation matrix M
    """
    img_size = img.shape[1::-1]
    M = cv2.getPerspectiveTransform(src, dst)

    return M


def warp_image(img, src=None, dst=None, M=None):
    """
    Performs perspective transformation on an image

    @param img: OpenCV BGR image
    @param src (optional): numpy array of 4 points that will represent a rectangle
        in the warped image (not needed if @param M is provided)
    @param dst (optional): numpy array of 4 points that form the rectangle to which
        the src points will be mapped (not needed if @param M is provided)

    @returns warped image
    """

    img_size = img.shape[1::-1]
    if M is None:
        assert src is not None and dst is not None
        M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M


def fit_polynomial(lefty, leftx, righty, rightx, draw=False, img_shape=None):
    """
    Fits a second grade polynomial given the lane pixels

    @param lefty: lane pixel indexes corresponding to y-axis of left lane
    @param leftx: lane pixel indexes corresponding to x-axis of left lane
    @param righty: lane pixel indexes corresponding to y-axis of right lane
    @param rightx: lane pixel indexes corresponding to x-axis of right lane
    @param draw (optional): whether to compute linspace for plotting
    @param img_shape (optional): img.shape to calculate linspace for plotting
        (must be specified if draw=True)
    
    @returns left_fit, right_fit: polynomial coefficients for each lane
    @returns plot_y, left_fit_x, right_fit_x: linspaces for plotting
    """

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    plot_y, left_fit_x, right_fit_x = None, None, None
    if draw:
        plot_y = np.linspace(0, img_shape[0]-1, img_shape[0])
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

    return left_fit, right_fit, plot_y, left_fit_x, right_fit_x


def fit_polynomial_with_sliding_window(binary_img, lane_detection=None, num_windows=9, margin=100, min_pix=50, draw=False, plot=False):
    """
    Finds the lane pixels in a warped image with sliding windows

    @param binary_img: one channel binary image range[0|1]    
    @param num_windows: number of windows to divide the image vertically
    @param margin: horizontal window margin (+- margin)
    @param min_pix: minimum number of pixels found to recenter window
    @param draw (optional): whether to return an output image with the
        windows rectangles

    @returns left_fit, right_fit: polynomial coefficients for each lane
    @return output_img: image with the windows rectangles(None if draw=False)
    """

    # Calculate horizontal histogram on the bottom half of the image to locate the lanes
    histogram = np.sum(binary_img[binary_img.shape[0]//2:,:], axis=0)

    # Calcuate de starting points of both lanes
    middle_point = np.int(histogram.shape[0]//2)
    left_x_idx = np.argmax(histogram[:middle_point])
    right_x_idx = np.argmax(histogram[middle_point:]) + middle_point

    # Caculate window height based on how many windows have been specified
    window_height = np.int(binary_img.shape[0]//num_windows)

    # Retrieve the indexes of all nonzero pixles in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Initialize arrays to save indexes of lane pixels
    left_px_idxs, right_px_idxs = [], []

    # Optional output image for visualization
    output_img = np.dstack((binary_img, binary_img, binary_img))*255 if draw else None

    for window_num in range(num_windows):

        # Define window boundaries
        top = binary_img.shape[0] - (window_num+1)*window_height
        bottom = binary_img.shape[0] - window_num*window_height
        left_window_boundaries = {
            'top': top,
            'right': left_x_idx + margin,
            'bottom': bottom,
            'left': left_x_idx - margin if left_x_idx - margin > 0 else 0
        }
        right_window_boundaries = {
            'top': top,
            'right': right_x_idx + margin if right_x_idx + margin < binary_img.shape[1] else binary_img.shape[1],
            'bottom': bottom,
            'left': right_x_idx - margin
        }

        # Visualize
        if draw:
            cv2.rectangle(output_img,
                          (left_window_boundaries['left'], left_window_boundaries['bottom']),
                          (left_window_boundaries['right'], left_window_boundaries['top']),
                          (255, 0, 0), 4)
            cv2.rectangle(output_img,
                          (right_window_boundaries['left'], right_window_boundaries['bottom']),
                          (right_window_boundaries['right'], right_window_boundaries['top']),
                          (255, 0, 0), 4)

        # Extract nonzero pixel indexes that are within the current window
        left_idxs = ((nonzeroy >= left_window_boundaries['top'])
                     & (nonzeroy < left_window_boundaries['bottom'])
                     & (nonzerox >= left_window_boundaries['left'])
                     & (nonzerox < left_window_boundaries['right'])
                    ).nonzero()[0]
        right_idxs = ((nonzeroy >= right_window_boundaries['top'])
                     & (nonzeroy < right_window_boundaries['bottom'])
                     & (nonzerox >= right_window_boundaries['left'])
                     & (nonzerox < right_window_boundaries['right'])
                    ).nonzero()[0]

        # Save these lanel pixel indexes
        left_px_idxs.append(left_idxs)
        right_px_idxs.append(right_idxs)

        # If enough pixels were found, recenter the window to the average x position
        if len(left_idxs) > min_pix:
            left_x_idx = np.int(np.mean(nonzerox[left_idxs]))
        if len(right_idxs) > min_pix:
            right_x_idx = np.int(np.mean(nonzerox[right_idxs]))

    left_px_idxs = np.concatenate(left_px_idxs)
    right_px_idxs = np.concatenate(right_px_idxs)

    # Extract left and right line pixel indexes
    lefty = nonzeroy[left_px_idxs]
    leftx = nonzerox[left_px_idxs]
    righty = nonzeroy[right_px_idxs]
    rightx = nonzerox[right_px_idxs]

    if draw:
        output_img[lefty, leftx] = [255, 213, 0]
        output_img[righty, rightx] = [0, 213, 255]

    left_fit, right_fit = None, None
    try:
        left_fit, right_fit, plot_y, left_fit_x, right_fit_x = fit_polynomial(lefty,
            leftx, righty, rightx, draw=draw, img_shape=binary_img.shape)

        if lane_detection is not None:
            lane_detection.left_fits.append(left_fit)
            lane_detection.right_fits.append(right_fit)

            left_fit = np.mean(lane_detection.left_fits, axis=0)
            right_fit = np.mean(lane_detection.right_fits, axis=0)

            left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
            right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

    except TypeError:
        # The polyfit function wasn't able to compute any coefficients
        print('The function failed to fit a lane!')
        plot_y = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
        left_fit_x = plot_y**2 + plot_y
        right_fit_x = plot_y**2 + plot_y

    if plot:
        # Plot the polynomial lines onto the image
        plt.plot(left_fit_x, plot_y, color='magenta')
        plt.plot(right_fit_x, plot_y, color='magenta')

    return left_fit, right_fit, output_img


def fit_polynomial_with_previous_coefficients(binary_img, left_fit, right_fit, lane_detection=None, margin=100, sliding_window_params=None, draw=False, plot=False):
    """
    Finds the lane pixels in a warped image with sliding windows

    @param binary_img: one channel binary image range[0|1]
    @param left_fit: polynomial coefficients for left lane
    @param right_fit: polynomial coefficients for right lane
    @param margin: margin in which to look for new pixels with respect
        to the previous polynomial
    @param sliding_window_params (optional): tuple containing the optional
        parameters for fit_polynomial_with_sliding_window
        (num_windows, margin, min_pix)
    @param draw (optional): whether to return an output image with the
        windows rectangles

    @returns left_fit, right_fit: polynomial coefficients for each lane
    @return output_img: image with the windows rectangles(None if draw=False)
    """

    # Optional output image for visualization
    output_img = np.dstack((binary_img, binary_img, binary_img))*255 if draw else None

    # Retrieve the indexes of all nonzero pixles in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Calculate indexes within the margin
    left_poly = (left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2])
    right_poly = (right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2])

    left_idxs = ((nonzerox > (left_poly - margin)) & (nonzerox < (left_poly + margin)))
    right_idxs = ((nonzerox > (right_poly - margin)) & (nonzerox < (right_poly + margin)))

    # Extract left and right line pixel indexes
    lefty = nonzeroy[left_idxs]
    leftx = nonzerox[left_idxs]
    righty = nonzeroy[right_idxs]
    rightx = nonzerox[right_idxs]

    if draw:
        output_img[lefty, leftx] = [255, 213, 0]
        output_img[righty, rightx] = [0, 213, 255]

    try:
        left_fit, right_fit, plot_y, left_fit_x, right_fit_x = fit_polynomial(lefty,
            leftx, righty, rightx, draw=draw, img_shape=binary_img.shape)

        if lane_detection is not None:
            lane_detection.left_fits.append(left_fit)
            lane_detection.right_fits.append(right_fit)        

            left_fit = np.mean(lane_detection.left_fits, axis=0)
            right_fit = np.mean(lane_detection.right_fits, axis=0)

            left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
            right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

    except TypeError:
        # The polyfit function wasn't able to compute any coefficients
        print('The function failed to fit a lane. Falling back to sliding windows.')
        if sliding_window_params is None:
            return fit_polynomial_with_sliding_window(binary_img, draw=draw)
        else:
            num_windows, sliding_window_margin, min_pix = sliding_window_params
            return fit_polynomial_with_sliding_window(binary_img, num_windows=num_windows,
                                                      margin=sliding_window_margin,
                                                      min_pix=min_pix, draw=draw)

    if draw:
        left_with_margin = np.hstack((
            np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))]),
            np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, plot_y])))])
            ))
        right_with_margin = np.hstack((
            np.array([np.transpose(np.vstack([right_fit_x - margin, plot_y]))]),
            np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, plot_y])))])
            ))
        left_center_shared = np.hstack((
            np.array([np.transpose(np.vstack([left_fit_x, plot_y]))]),
            np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, plot_y])))])
            ))
        right_center_shared = np.hstack((
            np.array([np.transpose(np.vstack([right_fit_x-+ margin, plot_y]))]),
            np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
            ))
        center = np.hstack((
            np.array([np.transpose(np.vstack([left_fit_x + margin, plot_y]))]),
            np.array([np.flipud(np.transpose(np.vstack([right_fit_x - margin, plot_y])))])
            ))

        # Draw the lane onto the warped blank image
        window_img = np.zeros_like(output_img)
        cv2.fillPoly(window_img, np.int_([left_with_margin]), (255, 0, 0))
        cv2.fillPoly(window_img, np.int_([right_with_margin]), (255, 0, 0))
        cv2.fillPoly(window_img, np.int_([left_center_shared]), (255, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_center_shared]), (255, 255, 0))
        cv2.fillPoly(window_img, np.int_([center]), (0, 255, 0))
        output_img = cv2.addWeighted(output_img, 1, window_img, 0.3, 0)

    if plot:
        # Plot the polynomial lines onto the image
        plt.plot(left_fit_x, plot_y, color='magenta')
        plt.plot(right_fit_x, plot_y, color='magenta')

    return left_fit, right_fit, output_img


def calculate_curve_radius(y_eval, left_fit, right_fit, ym_per_pix=30/720):
    """
    Calculate curve radius given polynomial coefficients

    @param y_eval: vertical point at which evalute the curvature. It could be the
        height of the image to get the radius at the closest point to the car.
    @param left_fit: polynomial coefficients for left lane
    @param right_fit: polynomial coefficients for right lane
    @param ym_per_pix: conversion in y from pixels space to meters

    @returns left_curve_radius, right_curve_radius: curve radiuses of both lanes
    """

    left_curve_radius = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curve_radius = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    avg_radius = np.mean([left_curve_radius, right_curve_radius])

    return left_curve_radius, right_curve_radius, avg_radius


class LaneDetection():
    """
    Class to store the last coefficients to search for new lanes

    @attr poly_fits: coefficients of lane polynomial fit
    @attr radiuses: last radiuses of curvature for averaging
    """

    def __init__(self, left_fits=[], right_fits=[], radiuses=[]):
        self.left_fits = left_fits
        self.right_fits = right_fits
        self.radiuses = radiuses


def video_pipeline(video_path, output_video_path, hyperparameters={}):
    """
    Detects lanes on a video frame

    @param video_path: path to video file in which to detect lanes on
    @param hyperparameters: dictionary containing hyperparameters for the pipeline
        {
            'undistort': {
                'mtx': mtx,
                'dist': dist
            },
            'saturation_brightness_min': 20,
            'abs_sobel_threshold_x': {
                'kernel_size': 3,
                'thresh': (20, 100)
            },
            'abs_sobel_threshold_y': {
                'kernel_size': 3,
                'thresh': (20, 100)
            },
            'mag_sobel_threshold': {
                'kernel_size': 3,
                'thresh': (30, 100)
            },
            'dir_sobel_threshold': {
                'kernel_size': 3,
                'thresh': (0.7, 1.3)
            },
            'saturation_threshold': {
                'thresh': (170, 255)
            },
            'warp': {
                'src': src,
                'dst': dst
            },
            'poly_sliding_window': {
                'num_windows': 9,
                'margin': 100,
                'min_pix': 50
            },
            'poly_previous': {
                'margin': 140
            },
            'curve_radius': {
                'ym_per_pix': 30/720,
                'xm_per_pix': 3.7/700
            },
            'avg_radius_num_frames': 1,
            'avg_poly_num_frames': 1
        }
    """

    lane_detection = LaneDetection()
    M_to_birds_eye_view, M_to_original = None, None

    def process_image(img, hyperparameters, lane_detection, M_to_birds_eye_view, M_to_original):
        """
        Detects lanes on a video frame

        @param img: RGB image
        @param hyperparameters: dictionary containing hyperparameters for the
            pipeline

        @returns RGB processed image
        """

        # Undistort image
        assert 'undistort' in hyperparameters and 'mtx' in hyperparameters['undistort']\
            and 'dist' in hyperparameters['undistort']
        undist = undistort_image(img, hyperparameters['undistort'])

        bgr = cv2.cvtColor(undist, cv2.COLOR_RGB2BGR)

        # Extracting saturation
        hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
        h, l, s = hls[:,:,0], hls[:,:,1], hls[:,:,2]
        if 'saturation_brightness_min' in hyperparameters:
            s[l < hyperparameters['saturation_brightness_min']] = 0
        else:
            saturation_brightness_min = 20
            s[l < saturation_brightness_min] = 0

        # Absolute sobel threshold x
        if 'abs_sobel_threshold_x' in hyperparameters:
            params = hyperparameters['abs_sobel_threshold_x']
            assert 'kernel_size' in params and 'thresh' in params
            gradx_binary = abs_sobel_threshold(bgr, 'x',
                                               sobel_kernel=params['kernel_size'],
                                               thresh=params['thresh'])
        else: gradx_binary = abs_sobel_threshold(bgr, 'x')


        # Absolute sobel threshold y
        if 'abs_sobel_threshold_y' in hyperparameters:
            params = hyperparameters['abs_sobel_threshold_y']
            assert 'kernel_size' in params and 'thresh' in params
            grady_binary = abs_sobel_threshold(bgr, 'y',
                                               sobel_kernel=params['kernel_size'],
                                               thresh=params['thresh'])
        else: grady_binary = abs_sobel_threshold(bgr, 'y')


        # Magnitude sobel threshold
        if 'mag_sobel_threshold' in hyperparameters:
            params = hyperparameters['mag_sobel_threshold']
            assert 'kernel_size' in params and 'thresh' in params
            mag_binary = mag_sobel_threshold(bgr,
                                             sobel_kernel=params['kernel_size'],
                                             thresh=params['thresh'])
        else: mag_binary = mag_sobel_threshold(bgr)


        # Direction sobel threshold
        if 'dir_sobel_threshold' in hyperparameters:
            params = hyperparameters['dir_sobel_threshold']
            assert 'kernel_size' in params and 'thresh' in params
            dir_binary = dir_sobel_threshold(bgr,
                                             sobel_kernel=params['kernel_size'],
                                             thresh=params['thresh'])
        else: dir_binary = dir_sobel_threshold(bgr)


        # Saturation threshold
        if 'saturation_threshold' in hyperparameters:
            params = hyperparameters['saturation_threshold']
            assert 'thresh' in params
            s_threshold = saturation_threshold(s, thresh=params['thresh'])
        else: s_threshold = saturation_threshold(s)

        h_threshold = hue_threshold(h)

        # Combined thresholds
        combined = np.zeros_like(undist[:,:,0])
        mask = (\
                ((gradx_binary == 1) & (grady_binary == 1))\
                 | \
                ((mag_binary == 1) & (dir_binary == 1))\
                 | \
                ((s_threshold == 1) & (h_threshold == 1))\
               )
        combined[mask] = 1

        # Warp image (bird's eye view)
        if M_to_birds_eye_view is None:
            assert 'warp' in hyperparameters and 'src' in hyperparameters['warp']\
                and 'dst' in hyperparameters['warp']
            warped, M_to_birds_eye_view = warp_image(combined,
                                                     hyperparameters['warp']['src'],
                                                     hyperparameters['warp']['dst'])
        else:
            warped, _ = warp_image(combined, M=M_to_birds_eye_view)

        # Fit polynomial
        if len(lane_detection.left_fits) == 0:
            if 'poly_sliding_window' in hyperparameters:
                params = hyperparameters['poly_sliding_window']
                assert 'num_windows' in params and 'margin' in params and 'min_pix' in params
                left_fit, right_fit, output_img = fit_polynomial_with_sliding_window(warped,
                                                                                     lane_detection=lane_detection,
                                                                                     num_windows=params['num_windows'],
                                                                                     margin=params['margin'],
                                                                                     min_pix=params['min_pix'],
                                                                                     draw=True)
            else:
                left_fit, right_fit, output_img = fit_polynomial_with_sliding_window(warped, draw=True)
        else:
            left_fit = np.mean(lane_detection.left_fits, axis=0)
            right_fit = np.mean(lane_detection.right_fits, axis=0)
            if 'poly_previous' in hyperparameters:
                params = hyperparameters['poly_previous']
                assert 'margin' in params
                left_fit, right_fit, output_img = fit_polynomial_with_previous_coefficients(warped,
                                                                                            left_fit,
                                                                                            right_fit,
                                                                                            lane_detection=lane_detection,
                                                                                            margin=params['margin'],
                                                                                            draw=True)
            else:
                sliding_window_params = None
                if 'poly_sliding_window' in hyperparameters:
                    params = hyperparameters['poly_sliding_window']
                    assert 'num_windows' in params and 'margin' in params and 'min_pix' in params
                    sliding_window_params = (params['num_windows'], params['margin'], params['min_pix'])
                left_fit, right_fit, output_img = fit_polynomial_with_previous_coefficients(warped,
                                                                                            left_fit,
                                                                                            right_fit,
                                                                                            lane_detection=lane_detection,
                                                                                            sliding_window_params=sliding_window_params,
                                                                                            draw=True)

        if 'avg_poly_num_frames' in hyperparameters:
            lane_detection.left_fits = lane_detection.left_fits[-hyperparameters['avg_poly_num_frames']:]
            lane_detection.right_fits = lane_detection.right_fits[-hyperparameters['avg_poly_num_frames']:]
        else:
            avg_poly_num_frames = 30
            lane_detection.left_fits = lane_detection.left_fits[-avg_poly_num_frames:]
            lane_detection.right_fits = lane_detection.right_fits[-avg_poly_num_frames:]

        # Calculate radius
        if 'curve_radius' in hyperparameters:
            params = hyperparameters['curve_radius']
            assert 'ym_per_pix' in params and 'xm_per_pix' in params
            left_curve_radius, right_curve_radius, lane_avg_radius = calculate_curve_radius(warped.shape[0], left_fit, right_fit,
                                                                                            ym_per_pix=params['ym_per_pix'])
        else:
            left_curve_radius, right_curve_radius, lane_avg_radius = calculate_curve_radius(warped.shape[0], left_fit, right_fit)

        # Warp the lane boundaries back to the original perspective
        if M_to_original is None:
            assert 'warp' in hyperparameters and 'src' in hyperparameters['warp']\
                and 'dst' in hyperparameters['warp']
            unwarped, M_to_original = warp_image(output_img, hyperparameters['warp']['dst'],
                                                             hyperparameters['warp']['src'])
        else:
            unwarped, _ = warp_image(output_img, M=M_to_original)

        # Blend original image with boundaries
        img_with_boundaries = cv2.addWeighted(undist, .6, unwarped, 1, 0)

        # Caculate position of vehicle with respect to the center of the lane
        left_base = left_fit[0]*img.shape[0]**2 + left_fit[1]*img.shape[0] + left_fit[2]
        right_base = right_fit[0]*img.shape[0]**2 + right_fit[1]*img.shape[0] + right_fit[2]
        center = (right_base + left_base) / 2
        if 'curve_radius' in hyperparameters:
            params = hyperparameters['curve_radius']
            assert 'ym_per_pix' in params and 'xm_per_pix' in params
            x_pos = (img.shape[1]//2 - center) * params['xm_per_pix'] # calculate x position and convert to real world meters
        else:
            xm_per_pix = 3.7/700
            x_pos = (img.shape[1]//2 - center) * xm_per_pix # calculate x position and convert to real world meters

        lane_detection.radiuses.append(lane_avg_radius)
        if 'avg_radius_num_frames' in hyperparameters:
            lane_detection.radiuses = lane_detection.radiuses[-hyperparameters['avg_radius_num_frames']:]
        else:
            avg_radius_num_frames = 30
            lane_detection.radiuses = lane_detection.radiuses[-avg_radius_num_frames:]

        avg_radius = np.mean(lane_detection.radiuses)
        if avg_radius > 7_000:
            cv2.putText(img_with_boundaries, "Straight line", (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        else:
            cv2.putText(img_with_boundaries, f"Radius of curvature: {avg_radius:.0f} m", (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        cv2.putText(img_with_boundaries, f"Vehicle is {x_pos:.2f}m {'left' if x_pos < 0 else 'right'} of center", (10, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

        return img_with_boundaries

    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))

    video = VideoFileClip(video_path)
    processed_video = video.fl_image(lambda image: process_image(image, hyperparameters, lane_detection,
                                                                 M_to_birds_eye_view, M_to_original))
    processed_video.write_videofile(output_video_path, audio=False)
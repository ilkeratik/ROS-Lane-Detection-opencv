import cv2 as cv
import numpy as np

def abs_sobel_thresh( img, orient='x', thresh=(0,255)):
        """
        Canny Edge Detection combines the sobel gradient for both x and y. 
        By breaking it apart into its components, we can produced a refined version of Canny edge detection.
        """
        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output

def mag_threshold( img, sobel_kernel=3, thresh=(0, 255)):
    """
    The function will filter based on a min/max magnitude for the gradient. 
    This function is looking at the combined xy gradient, but it could be altered to filter on the magnitude in a single direction, 
    or some linear combination of the two. 
    """
    # 1) Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the xy magnitude 
    mag = np.sqrt(x**2 + y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale = np.max(mag)/255
    eightbit = (mag/scale).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(eightbit)
    binary_output[(eightbit > thresh[0]) & (eightbit < thresh[1])] =1 
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(np.pi/5, np.pi/2)):
    """
    This function will filter based on the direction of the gradient. 
    For lane detection, we will be interested in vertical lines that are +/- some threshold 
    near pi/2
    """
    # 1) Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    x = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel))
    y = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel))
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(y, x)
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return binary_output

def hls_select(img, sthresh=(90, 255),lthresh=(90,255)):
    """
    The gradient filters above all convert the original image into grayscale and a lot of useful information is lost. 
    Lane lines are either yellow or white, and we can use that to our advantage trying to locate and track them. 
    The **H**ue **S**aturation **L**ightness color space will help. 
    In particular, the S channel of an HSL image retains a lot information about lane lines - especially when there are shadows on the road. 
    The Red channel of RGB also does a good job of creating binary images of lane lines. 
    """
    # 1) Convert to HLS color space
    hls_img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1])
                & (L > lthresh[0]) & (L <= lthresh[1])] = 1

    return binary_output
    # frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    # hsv_binary = cv.inRange(frame_HSV, (70, 100, 0), (169, 200, 130))
    # s_binary = cv.bitwise_and(img,img, mask= hsv_binary)
    # s_binary = s_binary[:,:,1]
    # return s_binary

def binary_pipeline(img):
    
    #img_copy = cv.GaussianBlur(img, (5, 5), 0)
    img_copy = np.copy(img)
    
    # color channels
    s_binary = hls_select(img_copy)
    #red_binary = red_select(img_copy, thresh=(200,255))
    
    # Sobel x
    x_binary = abs_sobel_thresh(img_copy,thresh=(100, 200))
    y_binary = abs_sobel_thresh(img_copy,thresh=(100, 200), orient='y')
    xy = cv.bitwise_or(x_binary, y_binary)
    #magnitude & direction
    mag_binary = mag_threshold(img_copy, thresh=(100,200))
    dir_binary = dir_threshold(img_copy)
    
    # Stack each channel
    gradient = np.zeros_like(s_binary)
    gradient[np.logical_and(xy , (np.logical_and(mag_binary, dir_binary)))] = 1
    final_binary = cv.bitwise_or(s_binary, gradient)

    kernel = np.ones((5, 5), np.uint8)
    final_binary = cv.morphologyEx(final_binary.astype(np.uint8), cv.MORPH_CLOSE, kernel)
    final_binary = cv.dilate(final_binary, kernel, iterations=1)
    return final_binary

def calculateContours(thresh_img, original_img):

        contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        # ---------------------------------------------------------------------------
        # Originally developed by OutOfTheBots
        # (https://www.youtube.com/channel/UCCX7z2JENOSZ1mraOpyVyzg/about)

        for cont in contours:

        # if len(contours) > 0:
            blackbox = cv.minAreaRect(cont)
            (x_min, y_min), (w_min, h_min), ang = blackbox
            if ang < -45: ang += 90
            if w_min < h_min and ang > 0: ang = (90-ang)*-1
            if w_min > h_min and ang < 0: ang = 90 + ang
            setpoint = thresh_img.shape[1]/2
            cte = -int(x_min - setpoint)
            angle = -int(ang)

            if True:
                box = cv.boxPoints(blackbox)
                box = np.int0(box)
                _ = cv.drawContours(original_img, [box], 0, (255,0,0),1)
                cv.line(original_img, (int(x_min), 0), (int(x_min), thresh_img.shape[0]), (0,0,255), 1)
        # ---------------------------------------------------------------------------
        else:
            cte = None
            angle = None

        return cte, angle, original_img
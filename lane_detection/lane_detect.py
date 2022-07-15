#!/usr/bin/env python

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image
import pickle
import os
import glob

#from pathlib import Path

# import rospy
# # ROS Image message
# from sensor_msgs.msg import Image
# import std_msgs.msg
# # ROS Image message -> OpenCV2 image converter
# from cv_bridge import CvBridge, CvBridgeError
# bridge = CvBridge()

# from prius_msgs.msg import Control
# from alkan_custom_messages.msg import pid_lane

class LaneDetection:
    
    def __init__(self):
        self.base_path = '/home/nvidia/marc/src/racecar/scripts'
        # self.base_path = str(Path(__file__).parent)
        # os.chdir(self.base_path)
        self.is_enabled = True
        rospy.init_node('image_listener', anonymous=True)

        self.pid_pub = rospy.Publisher('pid_lane_pub', pid_lane, queue_size=1)

        self.image_topic = "/front_camera/color/image_raw"
        self.image_topic_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.frame_count = 0
        self.window_search = True
        #self.pub_control = rospy.Publisher('prius_controls', Control, queue_size=1)
    def res_ld(self):
        self.is_enabled = False

    def re_init(self):
        self.is_enabled = True
    def cam_calibration(self):
        images = glob.glob('camera_cal/calibration*.jpg')
        print(self.base_path)
        
        # store chessboard coordinates
        chess_points = []
        # store points from transformed img
        image_points = []
        # board is 6 rows by 9 columns. each item is one (xyz) point 
        # remember, only care about inside points. that is why board is 9x6, not 10x7
        chess_point = np.zeros((9*6, 3), np.float32)
        # z stays zero. set xy to grid values
        chess_point[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        for image in images:
            img = mpimg.imread(image)
            # convert to grayscale
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            width = 672 #640
            height = 376 #480
            dim = (width, height)
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

            # returns boolean and coordinates
            success, corners = cv.findChessboardCorners(gray, (9,6), None)
            
            if success:
                image_points.append(corners)
                #these will all be the same since it's the same board
                chess_points.append(chess_point)
            else:
                print('corners not found {}'.format(image))
        
        image = mpimg.imread((self.base_path+"/" + 'camera_cal/calibration2.jpg'))

        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        # ax1.imshow(image)
        # ax1.set_title('Captured Image', fontsize=30)

        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) 
        ret , corners = cv.findChessboardCorners(gray,(9,6),None)    
        if ret == False:
            print('corners not found')
        img1 = cv.drawChessboardCorners(image,(9,6),corners,ret) 

        # ax2.imshow(img1)
        # ax2.set_title('Corners drawn Image', fontsize=30)
        # plt.tight_layout()
        # plt.savefig((self.base_path / 'saved_figures/chess_corners.png').resolve())
        # plt.show()

        # Save everything!
        img = mpimg.imread(images[0])
        points_pkl = {}
        points_pkl["chesspoints"] = chess_points
        points_pkl["imagepoints"] = image_points
        points_pkl["imagesize"] = (672, 376) #(640, 480)
        pickle.dump(points_pkl,open(self.base_path+"/object_and_image_points.pkl", "wb" ))

        points_pickle = pickle.load( open(self.base_path+ "/object_and_image_points.pkl", "rb" ) )
        chess_points = points_pickle["chesspoints"]
        image_points = points_pickle["imagepoints"]
        img_size = points_pickle["imagesize"]

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(chess_points, image_points, img_size, None, None)
        
        camera = {}
        camera["mtx"] = mtx
        camera["dist"] = dist
        camera["imagesize"] = img_size
        pickle.dump(camera, open(self.base_path+"/camera_matrix.pkl", "wb"))

    def distort_correct(self,img,mtx,dist,camera_img_size):
        img_size1 = (img.shape[1],img.shape[0])
        # print(img_size1, camera_img_size)
        assert (img_size1 == camera_img_size),'image size is not compatible'
        undist = cv.undistort(img, mtx, dist, None, mtx)
        return undist

    def test_disort_correct(self,mtx,dist,img_size):
        img = mpimg.imread('camera_cal/calibration2.jpg')
        img_size1 = (img.shape[1], img.shape[0])
        undist = self.distort_correct(img, mtx, dist, img_size)

        ### Visualize the captured 
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Captured Image', fontsize=30)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.tight_layout()
        plt.show()
        plt.savefig('saved_figures/undistorted_chess.png')

    def abs_sobel_thresh(self, img, orient='x', thresh=(0,255)):
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

    def mag_threshold(self, img, sobel_kernel=3, thresh=(0, 255)):
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

    def dir_threshold(self,img, sobel_kernel=3, thresh=(0, np.pi/2)):
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

    def hls_select(self,img, sthresh=(90, 255),lthresh=(90,255)):
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

    def binary_pipeline(self,img):
        
        #img_copy = cv.GaussianBlur(img, (5, 5), 0)
        img_copy = np.copy(img)
        
        # color channels
        s_binary = self.hls_select(img_copy)
        #red_binary = red_select(img_copy, thresh=(200,255))
        
        # Sobel x
        x_binary = self.abs_sobel_thresh(img_copy,thresh=(100, 200))
        y_binary = self.abs_sobel_thresh(img_copy,thresh=(100, 200), orient='y')
        xy = cv.bitwise_or(x_binary, y_binary)
        #magnitude & direction
        mag_binary = self.mag_threshold(img_copy, thresh=(100,200))
        dir_binary = self.dir_threshold(img_copy)
        
        # Stack each channel
        gradient = np.zeros_like(s_binary)
        gradient[np.logical_and(xy , (np.logical_and(mag_binary, dir_binary)))] = 1
        final_binary = cv.bitwise_or(s_binary, gradient)

        kernel = np.ones((20, 20), np.uint8)
        final_binary = cv.morphologyEx(final_binary.astype(np.uint8), cv.MORPH_CLOSE, kernel)
        return final_binary

    def warp_image(self, img):
        
        image_size = (img.shape[1], img.shape[0])
        x = img.shape[1]
        y = img.shape[0]

        #the "order" of points in the polygon you are defining does not matter
        #but they need to match the corresponding points in destination_points!
        source_points = np.float32([
        [0.117 * x, y],
        [(0.5 * x) - (x*0.078), (2/3)*y],
        [(0.5 * x) + (x*0.078), (2/3)*y],
        [x - (0.117 * x), y]
        ])

        #     #chicago footage
        #     source_points = np.float32([
        #                 [300, 720],
        #                 [500, 600],
        #                 [700, 600],
        #                 [850, 720]
        #                 ])
            
        #     destination_points = np.float32([
        #                 [200, 720],
        #                 [200, 200],
        #                 [1000, 200],
        #                 [1000, 720]
        #                 ])
        
        destination_points = np.float32([
        [0.25 * x, y],
        [0.25 * x, 0],
        [x - (0.25 * x), 0],
        [x - (0.25 * x), y]
        ])

        src = np.float32([[1000,700],    # br
                       [300, 700],    # bl
                       [490, 450],   # tl
                       [890, 450]])  # tr
        dst = np.float32([[x, y],       # br
                      [0, y],       # bl
                      [0, 0],       # tl
                      [x, 0]])      # tr

        src = np.float32([[640,320],    # br
                       [100, 320],    # bl
                       [200, 230],   # tl
                       [525, 230]])  # tr
        dst = np.float32([[x, y],       # br
                      [0, y],       # bl
                      [0, 0],       # tl
                      [x, 0]])      # tr
        # height, width = img.shape[0:2]
        # output_size = height/2
        # corners = np.float32([[44,560], [378,450],[902,450],[1215,560]])
        # corners = np.float32([[55,420], [255,260],[360,260],[560,420]])
        # new_top_left=np.array([corners[0,0],0])
        # new_top_right=np.array([corners[3,0],0])
        # xoffset = 0
        # offset=[xoffset,0]    
        # img_size = (img.shape[1], img.shape[0])
        # src = np.float32([corners[0],corners[1],corners[2],corners[3]])
        # dst = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset]) 
    
        M = cv.getPerspectiveTransform(src, dst)
        perspective_transform = cv.getPerspectiveTransform(src, dst)
        inverse_perspective_transform = cv.getPerspectiveTransform( dst, src)
        
        warped_img = cv.warpPerspective(img, perspective_transform, image_size, flags=cv.INTER_LINEAR)
        
        #print(source_points)
        #print(destination_points)
        
        return warped_img, inverse_perspective_transform

    def disort_img(self,image, img_loaded=False):
        camera = pickle.load(open(self.base_path+"/camera_matrix.pkl", "rb" ))
        mtx = camera['mtx']
        dist = camera['dist']
        camera_img_size = camera['imagesize']
        if not img_loaded:
            image = cv.imread(image)
        
        image = self.distort_correct(image,mtx,dist,camera_img_size)
        return image
    def right_left_lane_detect_test_plot(self):
        image = self.disort_img('framesVGA/7.png')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        f, ax = plt.subplots(3,3, figsize=(15,9))
        ax[0,0].set_title('Sobel-Canny Edge')
        ax[0,0].imshow(ld.abs_sobel_thresh(image, thresh=(20,110)), cmap='gray')

        ax[0,1].set_title('Magnitude based filter')
        mag = ld.mag_threshold(image, thresh=(20,100))
        ax[0,1].imshow(mag,  cmap='gray')

        ax[0,2].set_title('Direction of gradient')
        ax[0,2].imshow(ld.dir_threshold(image, thresh=(0.8,1.5)),  cmap='gray')

        ax[1,0].set_title('hls white filter')
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        x= cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        y = cv.Canny(gray,50,150,apertureSize=3)
        ax[1,0].imshow(x)
        
        res = ld.binary_pipeline(image)
        ax[1,1].set_title('Mixed filtering pipeline')
        ax[1,1].imshow(y,  cmap='gray')

        birdseye_result, inverse_perspective_transform = ld.warp_image(res)
        ax[1,2].set_title('Birdseye')
        ax[1,2].imshow(birdseye_result)
        splitted = (mag.shape[1]-600)/ 10
        points = [x*splitted+600 for x in range(1,10)]

        lines = cv.HoughLinesP(
            y, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=100, # Min number of votes for valid line
            minLineLength=50, # Min allowed length of line
            maxLineGap=5 # Max allowed gap between line for joining them
            )
 
        # Iterate over points
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            ax[2,0].plot((x1,y1),(x2,y2))
            # Maintain a simples lookup list for points

        # ax[2,0].imshow(mag[:500,600:])
        # print(points)
        # ax[0,1].axvline(x = 600, color = 'r', linestyle = '-')
        # for _,x in enumerate(points):
        #         ax[0,1].axvline(x = x, color = 'y', linestyle = '-')
        # histogram = np.split(mag[:500,600:],10, axis=1)
        # histogram = [x.sum() for x in histogram]
        # print(histogram)
        # ax[2,2].plot(histogram)
        plt.show()
        

    def test_plot(self):
        image = self.disort_img('framesVGA/19.png')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        img_copy = np.copy(image)
        f, ax = plt.subplots(3,3, figsize=(16,11))
        ax[0,0].set_title('Sobel-Canny Edge Filtresi')
        ax[0,0].imshow(self.abs_sobel_thresh(img_copy, thresh=(100,200), orient='y'), cmap='gray')
        
        img_copy = np.copy(image)
        ax[0,1].set_title('Büyüklük tabanlı filtre')
        mag = self.mag_threshold(img_copy, thresh=(100,200))
        ax[0,1].imshow(mag,  cmap='gray')

        img_copy = np.copy(image)
        ax[0,2].set_title('Gradyan yönü filtresi')
        ax[0,2].imshow(self.dir_threshold(img_copy),  cmap='gray')

        img_copy = np.copy(image)
        ax[1,0].set_title('HLS Uzayında beyaz renk filtresi')
        out_hsv = self.hls_select(img_copy)
        ax[1,0].imshow(out_hsv,  cmap='gray')

        img_copy = np.copy(image)
        res = self.binary_pipeline(img_copy)
        ax[1,1].set_title('Tüm filtrelerin birleşimi')
        ax[1,1].imshow(res,  cmap='gray')

        birdseye_result, inverse_perspective_transform = self.warp_image(res)
        ax[1,2].set_title('Kuşbakışına çevrilmiş görüntü')
        ax[1,2].imshow(birdseye_result)
        
        windows,l,r, r_state = self.track_lanes_initialize(birdseye_result)
        print(f'right_state: {r_state}')
        ax[2,0].set_title('Çerçeve yöntemi uygulanmış görüntü')
        ax[2,0].imshow(windows)

        x,y = image.shape[1], image.shape[0]
        temp = cv.warpPerspective(birdseye_result, inverse_perspective_transform, (x,y))
        ax[2,1].set_title('Şerit tespitinin görüntüye yansıtılması')
        
        ploty = np.linspace(0, birdseye_result.shape[0]-1, birdseye_result.shape[0])
        left_fitx = self.get_val(ploty,l)
        right_fitx = self.get_val(ploty,r)
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(birdseye_result).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast x and y for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane 
        cv.fillPoly(color_warp, np.int_([pts]), (255,0, 0))
        newwarp = cv.warpPerspective(color_warp, inverse_perspective_transform, (birdseye_result.shape[1], birdseye_result.shape[0])) 
        # overlay
        #newwarp = cv.cvtColor(newwarp, cv.COLOR_BGR2RGB)
        result = cv.addWeighted(image, 1, newwarp, 0.4, 0)
        ax[2,1].imshow(result)
        curve = self.measure_curve(birdseye_result,l,r)
        print(f'curve: {curve}')
        histogram = np.split(birdseye_result[int(birdseye_result.shape[0]/2):,:],8, axis=1)
        histogram = [x.sum() for x in histogram]
        print(f'histogram: {histogram}')
        ax[2,2].plot(histogram)
        ax[2,2].set_title("Pixel yoğunluğu grafiği")
        ax[2,2].ticklabel_format(useOffset=False, style='plain')
        print(f'offset: {self.vehicle_offset(windows,l,r)}')
        plt.show()
    
    def hist_plot(self):
        image = self.disort_img('frames/frame3.png')
        res = self.binary_pipeline(image)
        birdseye_result, inverse_perspective_transform = self.warp_image(res)

        histogram = np.sum(birdseye_result[int(birdseye_result.shape[0]/2):,:], axis=1)
        print(histogram)
        plt.figure()
        plt.plot(histogram)
        plt.show()
        plt.savefig('saved_figures/lane_histogram.png')

    def track_lanes_initialize(self, binary_warped):
        try:
        #   print(binary_warped.shape)
            global window_search
            print(binary_warped.shape)
            histogram = np.split(binary_warped[3*int(binary_warped.shape[0]/4):,:],16, axis=1)
            histogram = [x.sum() for x in histogram]
            # histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            # print(len(histogram))
            # print(histogram)
            # we need max for each half of the histogram. 
            left_state, right_state = True, True
            midpoint = np.int32(binary_warped.shape[1]/2)
            print(midpoint)
            leftx_base_idx = np.argmax(histogram[0:8]) 
            leftx_base = (leftx_base_idx+1) * 42
            rightx_base_idx = np.argmax(histogram[8:])
            rightx_base = (rightx_base_idx+1) * 42 +midpoint 
            print(histogram)
            print(histogram[leftx_base_idx], histogram[8+rightx_base_idx])
            print(leftx_base, rightx_base)
            if histogram[leftx_base_idx] <300:
                left_state = False
                leftx_base = 42

            if histogram[8+rightx_base_idx] <300:
                right_state = False 
                rightx_base = 670
            
            print(leftx_base_idx, rightx_base_idx)
            # Choose the number of sliding windows
            # this will throw an error in the height if it doesn't evenly divide the img height
            nwindows = 8
            # Set height of windows
            
            window_height = np.int32(round(binary_warped.shape[0]/nwindows))
            
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            
            # Set the width of the windows +/- margin
            margin = 50
            # Set minimum number of pixels found to recenter window
            minpix = 5
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []
            
            
            # Step through the windows one by one
            for window in range(nwindows):
            
                # Identify window boundaries in x and y (and right and left)
                win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
                win_y_high = int(binary_warped.shape[0] - window*window_height)
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
                cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[-1]

                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if left_state and len(good_left_inds) > minpix:
                    leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
                else:
                    leftx_current = leftx_current
                if right_state and len(good_right_inds) > minpix:        
                    rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
                else:
                    rightx_current = rightx_current

                    
            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            # print(left_lane_inds, right_lane_inds)
            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 
            
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 100
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except TypeError as e:
            print(e)
        return out_img,left_fit,right_fit, right_state
 
    def track_lanes_update(self, binary_warped, left_fit,right_fit):
        # repeat window search to maintain stability
        if self.frame_count % 5 == 0:
            self.window_search=True
        else:
            self.window_search =False
    
            
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


        return left_fit,right_fit,leftx,lefty,rightx,righty

    def get_val(self, y,poly_coeff):
        return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]

    def measure_curve(self,binary_warped,left_fit,right_fit):
        
        # generate y values 
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        
        # measure radius at the maximum y value, or bottom of the image
        # this is closest to the car 
        y_eval = np.max(ploty)
        
        # coversion rates for pixels to metric
        # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
        # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
        ym_per_pix = 5/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
        # x positions lanes
        leftx = self.get_val(ploty,left_fit)
        rightx = self.get_val(ploty,right_fit)

        # fit polynomials in metric 
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        
        # calculate radii in metric from radius of curvature formula
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        # averaged radius of curvature of left and right in real world space
        # should represent approximately the center of the road
        curve_rad = round((left_curverad + right_curverad)/2)
        
        return curve_rad

    def vehicle_offset(self, img,left_fit,right_fit):
    
        # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
        # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
        xm_per_pix = 0.9/ 376 #480
        image_center = img.shape[1]/2
        
        ## find where lines hit the bottom of the image, closest to the car
        left_low = self.get_val(img.shape[0],left_fit)
        right_low = self.get_val(img.shape[0],right_fit)
        
        # pixel coordinate for center of lane
        lane_center = (left_low+right_low)/2.0
        
        ## vehicle offset
        distance = image_center - lane_center
        
        ## convert to metric
        return (round(distance*xm_per_pix,5))

    def process_pipeline(self, frame, keep_state=False):
        frame = self.disort_img(frame, img_loaded=True)
        # print(frame.shape)
        dir_tresh = self.dir_threshold(frame, sobel_kernel=3, thresh=(0.8, 1.8))
        res = self.binary_pipeline(frame)
        birdseye_result, inverse_perspective_transform = self.warp_image(res)
        if self.frame_count > 0:
            l = self.last_l 
            r = self.last_r
            out = self.last_out
        if self.window_search:
            print(self.window_search, 'init')
            out, l,r,r_state = self.track_lanes_initialize(birdseye_result)
            offset = self.vehicle_offset(out, l, r)
            if not r_state:
                print('no right lane')
                offset = 1.0
            else:
                offset = self.vehicle_offset(out, l, r)

            self.window_search = False
        else:
            print('up')
            l,r, leftx,lefty,rightx,righty  = self.track_lanes_update(birdseye_result,l,r)
        # print(ld.measure_curve(birdseye_result,l, r))

        print(offset)  

        blend_on_road = frame
        
        try:
            # mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(blend_on_road, 'Processed frame count: {}'.format(self.frame_count), (450, 40), font, 0.4, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset), (450, 100), font, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        except TypeError:
            print("Somewhere is broken")

        self.frame_count += 1
        self.last_l = l
        self.last_r = r
        self.last_out = out
        return blend_on_road, offset


    def image_callback(self,msg):
        # Convert your ROS Image message to OpenCV2
        if self.is_enabled:
            try:
                frame = bridge.imgmsg_to_cv2(msg, "bgr8")
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                # width = 1280
                # height = 720
                # dim = (width, height)
                # frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
                try:
                    blend, offset = self.process_pipeline(frame, keep_state=False)#
                    cv.imshow("result",cv.cvtColor(blend, code=cv.COLOR_BGR2RGB))
                    cv.waitKey(1)
                    if offset > 15 or offset < -15:
                        pass
                    else:
                        
                        h = std_msgs.msg.Header()
                        h.stamp = rospy.Time.now()
                        command = pid_lane()
                        command.header = h
                        command.offset = offset
                        self.pid_pub.publish(command)
                    
                except TypeError as e:
                    print("message ",e)
                
            except CvBridgeError as e:
                print(e)

    def calle(self, msg):
        msg = bridge.imgmsg_to_cv2(msg, "bgr8")
        try:

            img_copy = cv.GaussianBlur(msg, (5, 5), 0)
            #img_copy = np.copy(img)
            
            # color channels
            s_binary = self.hls_select(img_copy)
            #print(s_binary.shape)
            
            # Sobel x
            x_binary = self.abs_sobel_thresh(img_copy,thresh=(25, 200))
            y_binary = self.abs_sobel_thresh(img_copy,thresh=(25, 200), orient='y')
            xy = cv.bitwise_and(x_binary, y_binary)
            
            #magnitude & direction
            mag_binary = self.mag_threshold(img_copy, sobel_kernel=3, thresh=(50,100))
            dir_binary = self.dir_threshold(img_copy, sobel_kernel=3, thresh=(0.8, 1.2))
            
            # Stack each channel
            gradient = np.zeros_like(s_binary)
            gradient[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
            final_binary = cv.bitwise_or(s_binary, gradient)

            kernel = np.ones((20, 20), np.uint8)
            final_binary = cv.morphologyEx(final_binary.astype(np.uint8), cv.MORPH_CLOSE, kernel)
            cv.imshow("result",final_binary)
            cv.waitKey(10)
        except TypeError as e:
            print("message ",e)

if __name__ == "__main__":
    # load camera matrix and distortion matrix
    ld = LaneDetection()
    if not os.path.exists(ld.base_path+"/camera_matrix.pkl"):
        ld.cam_calibration()


    ld.test_plot()

    # try:
    #    rospy.spin()
    # except KeyboardInterrupt:
    #    print("Shutting down")
    # cv.destroyAllWindows()



# h, w = blend_on_road.shape[:2]
        # thumb_ratio = 0.2
        # thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

        # off_x, off_y = 20, 15
        # mask = blend_on_road.copy()
        # mask = cv.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv.FILLED)
        # blend_on_road = cv.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

        # add thumbnail of binary image
        # thumb_binary = cv.resize(dir_tresh, dsize=(thumb_w, thumb_h))
        # thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
        # blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

        # add thumbnail of bird's eye view
        # thumb_birdeye = cv.resize(birdseye_result, dsize=(thumb_w, thumb_h))
        # thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
        # blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

        #add thumbnail of bird's eye view (lane-line highlighted)
        #thumb_img_fit = cv.resize(out, dsize=(thumb_w, thumb_h))
        # blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

        # add text (curvature and offset info) on the upper right of the blend
    
    

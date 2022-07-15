import cv2 as cv
import numpy as np


def lane_status_FSM(binary_warped):
    state = 'none'
    left_lane, right_lane = True, True
    histogram = np.split(binary_warped,16, axis=1)
    histogram = [x.sum() for x in histogram]
    midpoint = np.int32(binary_warped.shape[1]/2)
    leftx_base_idx = np.argmax(histogram[0:7]) 
    rightx_base_idx = np.argmax(histogram[8:])

    print(histogram)
    print(histogram[leftx_base_idx], histogram[8+rightx_base_idx])
    
    ## Pixel sanity check
    if histogram[leftx_base_idx] < 600 and histogram[8+rightx_base_idx] < 600:
        left_lane, right_lane = False, False

    elif histogram[leftx_base_idx] <600:
        left_lane = True
        right_lane = False # right lane is not detected
    elif histogram[8+rightx_base_idx] <600:
        right_lane = True
        left_lane = False # left lane is not detected

    # Both lanes are detected
    if left_lane and right_lane:
        state = 'both'
        leftx_base = (leftx_base_idx+1) * 42
        rightx_base = (rightx_base_idx+2) * 42 +midpoint
        out_im, line_fit, right_fit = track_lanes_initialize(binary_warped, leftx_base, rightx_base)
        return out_im, line_fit, right_fit

    # Only left lane is detected
    elif left_lane:
        state = 'left'
        base_idx = np.argmax(histogram)
        line_base = (base_idx+1) * 42
        out_im, line_fit = track_one_line(binary_warped, line_base)
        return out_im, line_fit, False

    # Only right lane is detected
    elif right_lane:
        state = 'right'
        base_idx = np.argmax(histogram)
        line_base = (base_idx+1) * 42
        out_im, right_fit = track_one_line(binary_warped, line_base)
        return out_im, False, right_fit

    print('current_st: '+state)

    return False, False, False

def track_one_line(binary_warped, line_base):
    try:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Choose the number of sliding windows
        # this will throw an error in the height if it doesn't evenly divide the img height
        nwindows = 8
        # Set height of windows
        window_height = np.int32(round(binary_warped.shape[0]/nwindows))
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current position to be updated for each window
        linex_current = line_base
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix =100
        # Create empty lists to receive left and right lane pixel indices
        line_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
        
            # Identify window boundaries in x and y (and right and left)
            win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
            win_y_high = int(binary_warped.shape[0] - window*window_height)
            win_xline_low = linex_current - margin
            win_xline_high = linex_current + margin
            # Identify the nonzero pixels in x and y within the window
            cv.rectangle(out_img,(win_xline_low,win_y_low),(win_xline_high,win_y_high),(0,255,0), 3) 
            good_line_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xline_low) & (nonzerox < win_xline_high)).nonzero()[0]
            line_inds.append(good_line_inds)
        
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_line_inds) > minpix:
                linex_current = np.int32(np.mean(nonzerox[good_line_inds]))

        # Concatenate the arrays of indices
        line_inds = np.concatenate(line_inds)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

        if len(line_inds) > 0:
            linex = nonzerox[line_inds]
            liney = nonzeroy[line_inds]
            line_fit = np.polyfit(liney, linex, 2)
            line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 50
        line_inds = ((nonzerox > (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] - margin)) & (nonzerox < (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] + margin))) 

        # Again, extract left and right line pixel positions
        linex = nonzerox[line_inds]
        liney = nonzeroy[line_inds]
        # Fit a second order polynomial to each
        line_fit = np.polyfit(liney, linex, 2)
    except TypeError as e:
            print(e)

    return out_img, line_fit

def track_lanes_initialize(binary_warped, leftx_base, rightx_base):
        try:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
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
            minpix =100
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
                
                # Identify the nonzero pixels in x and y within the window
                cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
            
                cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                right_lane_inds.append(good_right_inds)
                
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            
            print(f'len_left: {len(left_lane_inds)}, len right: {len(right_lane_inds)}')

            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

            # Extract left and right line pixel positions
            # Fit a second order polynomial to each
            # Generate x and y values for plotting
            if len(left_lane_inds) > 0:
                leftx = nonzerox[left_lane_inds]
                lefty = nonzeroy[left_lane_inds]
                line_fit = np.polyfit(lefty, leftx, 2)
                print(type(line_fit))
                line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]
            
            if len(right_lane_inds) > 0:
                rightx = nonzerox[right_lane_inds]
                righty = nonzeroy[right_lane_inds]
                right_fit = np.polyfit(righty, rightx, 2)
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 50
            left_lane_inds = ((nonzerox > (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] - margin)) & (nonzerox < (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] + margin))) 
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            line_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
                
            if type(line_fit) != np.ndarray:
                print('no proper left line')
                line_fit = False
            if type(right_fit) != np.ndarray:
                print('no proper right line')
                right_fit = False
        except TypeError as e:
                print(e)
        return out_img, line_fit, right_fit
 
def track_lanes_update(binary_warped, line_fit,right_fit):
    # repeat window search to maintain stability
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] - margin)) & (nonzerox < (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    line_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return line_fit,right_fit,leftx,lefty,rightx,righty

def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]
import numpy as np

def measure_curve(binary_warped,left_fit,right_fit):
        """
        high curveture radius means smooth curve-rotation ahead 
        """
        # generate y values 
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        
        # measure radius at the maximum y value, or bottom of the image
        # this is closest to the car 
        y_eval = np.max(ploty)
        
        # coversion rates for pixels to metric
        # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
        # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
        ym_per_pix = 10/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
        # x positions lanes
        leftx = get_val(ploty,left_fit)
        rightx = get_val(ploty,right_fit)

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

def measure_curve_one_line(binary_warped,line):
    """
        line: coefs of polyfit
    """
    # generate y values 
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # measure radius at the maximum y value, or bottom of the image
    # this is closest to the car 
    y_eval = np.max(ploty)
    
    # coversion rates for pixels to metric
    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    ym_per_pix = 10/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    linex = get_val(ploty, line)

    # fit polynomials in metric 
    line_cr = np.polyfit(ploty*ym_per_pix, linex*xm_per_pix, 2)
    
    # calculate radii in metric from radius of curvature formula
    line_curverad = ((1 + (2*line_cr[0]*y_eval*ym_per_pix + line_cr[1])**2)**1.5) / np.absolute(2*line_cr[0])
    
    return line_curverad

def vehicle_offset(img,left_fit,right_fit):

    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    xm_per_pix = 0.9/ 376 #480
    #image_center = img.shape[1]/2
    #right camera center
    image_center = 232
    #left camera center
    # image_center += 35
    ## vehicle offset

    ## find where lines hit the bottom of the image, closest to the car
    left_low = get_val(img.shape[0],left_fit)
    right_low = get_val(img.shape[0],right_fit)
    
    # pixel coordinate for center of lane
    lane_center = (left_low+right_low)/2.0
    
    distance = image_center - lane_center
    
    ## convert to metric
    return -1*(round(distance*xm_per_pix, 4))

def vehicle_offset_left_line(img, left_fit):

    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    xm_per_pix = 0.9/ 376 #480
    #image_center = (img.shape[1]/8)*2
    #right camera center
    image_center = 180
    #left camera center
    # image_center = 432
    ## vehicle offset

    ## find where lines hit the bottom of the image, closest to the car
    left_low = get_val(img.shape[0],left_fit)
    
    # pixel coordinate for center of lane
    lane_center = left_low
    distance = image_center - lane_center
    
    ## convert to metric
    return -1*(round(distance*xm_per_pix, 4))

def vehicle_offset_right_line(img, right_fit):

    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    xm_per_pix = 0.9/ 376 #480
    #image_center = (img.shape[1]/8)*6
    #right camera center
    image_center = 284
    #left camera center
    # image_center = 523
    ## find where lines hit the bottom of the image, closest to the car
    left_low = get_val(img.shape[0],right_fit)
    
    # pixel coordinate for center of lane
    lane_center = left_low
    distance = image_center - lane_center
    
    ## convert to metric
    return -1*(round(distance*xm_per_pix, 4))

def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]

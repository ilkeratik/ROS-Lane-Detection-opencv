import cv2 as cv
import numpy as np

def warp_image(img):
    
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

    # src = np.float32([[1000,700],    # br
    #                [300, 700],    # bl
    #                [490, 450],   # tl
    #                [890, 450]])  # tr
    # dst = np.float32([[x, y],       # br
    #               [0, y],       # bl
    #               [0, 0],       # tl
    #               [x, 0]])      # tr

    src = np.float32([[670,310],    # br
                [80, 310],    # bl
                [210, 240],   # tl
                [520, 240]])  # tr
    dst = np.float32([[x, y],       # br
                [0, y],       # bl
                [0, 0],       # tl
                [x, 0]])      # tr
    # dst = np.float32([
    #     [x - (0.25 * x), y],#br
    #     [x - (0.25 * x), 0], #bl
    #     [0.25 * x, 0], #tl
    #     [0.25 * x, y] #tr

    # ])
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
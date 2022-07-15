import pickle
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
from os import getcwd
base_path = getcwd()
# base_path ='/home/nvidia/marc/src/racecar/scripts'
def cam_calibration():
    images = glob.glob('camera_cal/calibration*.jpg')
    print(base_path)
    
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
    
    image = mpimg.imread((base_path + '/camera_cal/calibration2.jpg'))
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) 
    ret , corners = cv.findChessboardCorners(gray,(9,6),None)    
    if ret == False:
        print('corners not found')

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    # ax1.imshow(image)
    # ax1.set_title('Captured Image', fontsize=30)
    # img1 = cv.drawChessboardCorners(image,(9,6),corners,ret) 
    # ax2.imshow(img1)
    # ax2.set_title('Corners drawn Image', fontsize=30)
    # plt.tight_layout()
    # plt.savefig((base_path / 'saved_figures/chess_corners.png').resolve())
    # plt.show()

    # Save everything!
    img = mpimg.imread(images[0])
    points_pkl = {}
    points_pkl["chesspoints"] = chess_points
    points_pkl["imagepoints"] = image_points
    points_pkl["imagesize"] = (672, 376) #(640, 480)
    pickle.dump(points_pkl,open(base_path+"/object_and_image_points.pkl", "wb" ))

    points_pickle = pickle.load( open(base_path+ "/object_and_image_points.pkl", "rb" ) )
    chess_points = points_pickle["chesspoints"]
    image_points = points_pickle["imagepoints"]
    img_size = points_pickle["imagesize"]

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(chess_points, image_points, img_size, None, None)
    
    camera = {}
    camera["mtx"] = mtx
    camera["dist"] = dist
    camera["imagesize"] = img_size
    pickle.dump(camera, open(base_path+"/camera_matrix.pkl", "wb"))

def distort_correct(img,mtx,dist,camera_img_size):
    img_size1 = (img.shape[1],img.shape[0])
    # print(img_size1, camera_img_size)
    assert (img_size1 == camera_img_size),'image size is not compatible'
    undist = cv.undistort(img, mtx, dist, None, mtx)
    return undist

def test_disort_correct(mtx,dist,img_size):
    img = mpimg.imread('camera_cal/calibration2.jpg')
    img_size1 = (img.shape[1], img.shape[0])
    undist = distort_correct(img, mtx, dist, img_size)

    ### Visualize the captured 
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Captured Image', fontsize=30)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.tight_layout()
    plt.show()
    plt.savefig('saved_figures/undistorted_chess.png')


def undisort_img(image, img_loaded=True):
    camera = pickle.load(open(base_path+"/camera_matrix.pkl", "rb" ))
    mtx = camera['mtx']
    dist = camera['dist']
    camera_img_size = camera['imagesize']
    if not img_loaded:
        image = cv.imread(image)
    
    image = distort_correct(image,mtx,dist,camera_img_size)
    return image
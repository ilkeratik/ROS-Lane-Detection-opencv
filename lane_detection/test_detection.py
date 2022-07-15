from lane_detection.binary_utils import *
from lane_detection.camera_calibration import *
from lane_detection.image_perspective import *
from lane_detection.line_calculations import *
from lane_detection.sliding_windows import *

import matplotlib.pyplot as plt
# from os import getcwd
# base_path = getcwd()
def test_plot():
    image = cv.imread('framesVGA/7.png')
    image = undisort_img(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    img_copy = np.copy(image)
    f, ax = plt.subplots(3,3, figsize=(16,11))
    ax[0,0].set_title('Sobel-Canny Edge Filtresi')
    ax[0,0].imshow(abs_sobel_thresh(img_copy, thresh=(100,200), orient='y'), cmap='gray')
    
    img_copy = np.copy(image)
    ax[0,1].set_title('Büyüklük tabanlı filtre')
    mag = mag_threshold(img_copy, thresh=(100,200))
    ax[0,1].imshow(mag,  cmap='gray')

    img_copy = np.copy(image)
    ax[0,2].set_title('Gradyan yönü filtresi')
    ax[0,2].imshow(dir_threshold(img_copy),  cmap='gray')

    img_copy = np.copy(image)
    ax[1,0].set_title('HLS Uzayında beyaz renk filtresi')
    out_hsv = hls_select(img_copy)
    ax[1,0].imshow(out_hsv,  cmap='gray')

    img_copy = np.copy(image)
    res = binary_pipeline(img_copy)
    ax[1,1].set_title('Tüm filtrelerin birleşimi')
    ax[1,1].imshow(res,  cmap='gray')

    birdseye_result, inverse_perspective_transform = warp_image(res)
    ax[1,2].set_title('Kuşbakışına çevrilmiş görüntü')
    ax[1,2].imshow(birdseye_result)
    
    windows,l,r, state_changed= track_lanes_initialize(birdseye_result)
    
    ax[2,0].set_title('Çerçeve yöntemi uygulanmış görüntü')
    ax[2,0].imshow(windows)
    print(f'right_state: {r} , left_state: {l}')

    x,y = image.shape[1], image.shape[0]
    temp = cv.warpPerspective(birdseye_result, inverse_perspective_transform, (x,y))
    # ax[2,1].set_title('Şerit tespitinin görüntüye yansıtılması')
    
    
    # ploty = np.linspace(0, birdseye_result.shape[0]-1, birdseye_result.shape[0])
    # left_fitx = get_val(ploty,l)
    # right_fitx = get_val(ploty,r)
    
    # # Create an image to draw the lines on
    # warp_zero = np.zeros_like(birdseye_result).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # # Recast x and y for cv2.fillPoly()
    # pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    # pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # pts = np.hstack((pts_left, pts_right))
    
    # # Draw the lane 
    # cv.fillPoly(color_warp, np.int_([pts]), (255,0, 0))
    # newwarp = cv.warpPerspective(color_warp, inverse_perspective_transform, (birdseye_result.shape[1], birdseye_result.shape[0])) 
    # # overlay
    # #newwarp = cv.cvtColor(newwarp, cv.COLOR_BGR2RGB)
    # result = cv.addWeighted(image, 1, newwarp, 0.4, 0)

    # cte, angle, original_img= calculateContours(birdseye_result, birdseye_result)
    # print(f'cte: {cte}, angle: {angle}')
    # ax[2,1].imshow(result)

    histogram = np.split(birdseye_result[int(birdseye_result.shape[0]/2):,:],8, axis=1)
    histogram = [x.sum() for x in histogram]
    print(f'histogram: {histogram}')
    ax[2,2].plot(histogram)
    ax[2,2].set_title("Pixel yoğunluğu grafiği")
    ax[2,2].ticklabel_format(useOffset=False, style='plain')

    if  type(l) == np.ndarray and type(r) == np.ndarray:
        curve = measure_curve(birdseye_result,l,r)
        
        print(f'offset: {vehicle_offset(windows,l,r)}')
        print(f'curve: {curve}')

    plt.show()
    
def hist_plot():
    image = undisort_img('frames/frame40.png')
    res = binary_pipeline(image)
    birdseye_result, inverse_perspective_transform = warp_image(res)

    histogram = np.sum(birdseye_result[int(birdseye_result.shape[0]/2):,:], axis=1)
    print(histogram)
    plt.figure()
    plt.plot(histogram)
    plt.show()
    plt.savefig('saved_figures/lane_histogram.png')

if __name__ == '__main__':
    test_plot()
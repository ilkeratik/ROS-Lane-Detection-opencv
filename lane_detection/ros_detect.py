#!/usr/bin/env python

from racecar.msg import pid_lane
from binary_utils import *
from sliding_windows import *
from line_calculations import *
from image_perspective import *
from camera_calibration import *

import numpy as np
import cv2 as cv
from IPython.display import Image
import os
from os import getcwd
import rospy
# ROS Image message
from sensor_msgs.msg import Image
import std_msgs.msg
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

# from prius_msgs.msg import Control
# from alkan_custom_messages.msg import pid_lane
base_path = getcwd()


class LaneDetection:
    def __init__(self):
        # self.base_path = '/home/nvidia/marc/src/racecar/scripts/lane'
        self.base_path = getcwd()
        # self.base_path = str(Path(__file__).parent)
        # os.chdir(self.base_path)
        self.is_enabled = True
        rospy.init_node('lane_detection', anonymous=True)
        self.last_offset =0.0
        self.pid_pub = rospy.Publisher('pid_lane_pub', pid_lane, queue_size=1)

        self.image_topic = "/zed/left/image_rect_color"
        self.image_topic_sub = None
        self.frame_count = 0
        self.window_search = True

        # self.pub_control = rospy.Publisher('prius_controls', Control, queue_size=1)

    def res_ld(self):
        self.is_enabled = False

    def re_init(self):
        self.is_enabled = True

    def process_pipeline(self, frame, keep_state=False):
        out = None
        # print(frame.shape)
        res = binary_pipeline(frame)
        birdseye_result, inverse_perspective_transform = warp_image(res)

        if True: #self.window_search:
            # print(self.window_search, 'init')
            out, l,r = lane_status_FSM(birdseye_result)

        if type(l) != np.ndarray and type(l) != np.ndarray:
            print('no lanes found')
        if self.last_offset < 0:
            offset=self.last_offset + 0.3
        elif self.last_offset > 0:
            offset=self.last_offset - 0.3
        elif type(r) != np.ndarray:
            print('no right lane')
            # curve_rad = measure_curve_one_line(birdseye_result, l)
            # offset = vehicle_offset_left_line(birdseye_result, l)
            # offset = offset *(10.0/curve_rad)
            offset = 4.9

        elif type(l) != np.ndarray:
            print('no left lane')
            # curve_rad = measure_curve_one_line(birdseye_result, r)
            # offset = vehicle_offset_right_line(birdseye_result, r)
            # offset = -1.0 *offset *(10.0/curve_rad)
            offset = -4.9
        else:
            offset = vehicle_offset(out, l, r) *3
            if abs(offset) < 0.03:
                offset = 0.0


        self.window_search = False
        # else:
        #     print('up')
        #     l,r, leftx,lefty,rightx,righty  = track_lanes_update(birdseye_result,l,r)

        # font = cv.FONT_HERSHEY_SIMPLEX
        # cv.putText(out, 'Processed frame count: {}'.format(self.frame_count), (450, 40), font, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        # cv.putText(out, 'Offset from center: {:.02f}m'.format(offset), (450, 100), font, 0.4, (255, 255, 255), 1, cv.LINE_AA)

        print(offset)
        self.frame_count += 1
        self.last_l = l
        self.last_r = r
        self.last_offset = offset
        if type(out) != np.ndarray:
            out = frame
            self.last_out = out
        return out, offset

    def action_loop(self):
        try:
            while not rospy.is_shutdown():
                msg = rospy.wait_for_message(self.image_topic, Image)
                frame = bridge.imgmsg_to_cv2(msg, "rgb8")
                frame = undisort_img(frame)
                blend, offset= self.process_pipeline(frame, keep_state=False)
                cv.imshow("result",cv.cvtColor(blend, code=cv.COLOR_BGR2RGB))
                cv.waitKey(1)
                if offset > 10 or offset < -10:
                    pass
                else:
                    h = std_msgs.msg.Header()
                    h.stamp = rospy.Time.now()
                    command = pid_lane()
                    command.header = h
                    command.offset = offset
                    self.pid_pub.publish(command)

        except rospy.ROSInterruptException:
            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()
            command = pid_lane()
            command.header = h
            command.offset = 0.0
            self.pid_pub.publish(command)
            print('Detection terminated!')


    def image_callback(self,msg):
        # Convert your ROS Image message to OpenCV2
        if self.is_enabled:
            try:
                frame = bridge.imgmsg_to_cv2(msg, "rgb8")
                frame = undisort_img(frame)
                # width = 1280
                # height = 720
                # dim = (width, height)
                # frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
                try:
                    blend, offset= self.process_pipeline(frame, keep_state=False)#
                    cv.imshow("result",cv.cvtColor(blend, code=cv.COLOR_BGR2RGB))
                    cv.waitKey(1)

                    if offset > 10 or offset < -10:
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
            hlss = hls_select(msg)
            birdseye_result, inverse_perspective_transform = warp_image(msg)
            cv.imshow("result",hlss)
            cv.waitKey(1)
        except TypeError as e:
            print("message ",e)

if __name__ == "__main__":
    # load camera matrix and distortion matrix
    ld = LaneDetection()
    if not os.path.exists(ld.base_path+"/camera_matrix.pkl"):
        print("couldn't find calibration matrix...")
        cam_calibration()
    else:
        print("found calibration matrix...")

    ld.action_loop()
    # ld.image_topic_sub = rospy.Subscriber(ld.image_topic, Image, ld.image_callback)
    # try:
    #    rospy.spin()
    # except KeyboardInterrupt:
    #    print("Shutting down")
    # cv.destroyAllWindows()
    
    



# blend_on_road = frame
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

# add thumbnail of bird's eye view (lane-line highlighted)
# thumb_img_fit = cv.resize(out, dsize=(thumb_w, thumb_h))
# blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

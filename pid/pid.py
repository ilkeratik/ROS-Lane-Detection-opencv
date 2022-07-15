import rospy
from racecar.msg import pid_lane
from ackermann_msgs.msg import AckermannDriveStamped
from rospy import Duration
class SteeringCalculate:

    def __init__(self, ref, weights, is_discrete=False, sampling_time=1):
        """
        ref: desired error e.g 0 for 0 error
        weights: array of [KP, KI, KD] values
        is_discrete: discrete steps or continuous diff based steps
        sampling_time: if discrete, time diff for steps
        """
        self.ref = ref
        self.weights = weights
        self.is_discrete = is_discrete
        self.sampling_time = sampling_time
        rospy.init_node('steering_converter', anonymous=True)
        self.speed = 0.2
        self.pid_sub = rospy.Subscriber('pid_lane_pub', pid_lane, self.new_feedback)
        self.pub_command = rospy.Publisher('ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
        self.last_time_diff = Duration()
        self.is_first = True
        self.previous_message = None
        self.command = None
        self.clear()

    def clear(self):
        """
        A:  cumulative integral error
        """
        self.A = 0
        self.last_err = 0

    def new_feedback(self, msg):
        
        if self.is_first:
            # time_diff = 0.1
            self.previous_message = msg
            self.is_first = False
            self.offsets.append(msg.offset)
            time_diff=Duration()
        else:
            time_diff = msg.header.stamp - self.previous_message.header.stamp
            self.offsets.append(msg.offset)
            # time_diff = time_diff.to_sec()-self.last_time_diff
            # print(time_diff)
        if time_diff.to_sec()>=1:
            mean_offset = sum(self.offsets)/len(self.offsets)
            self.pid_step(mean_offset)
            self.offsets = []
            # self.last_time_diff = time_diff
            self.is_first=True
            self.previous_message = msg

    def pid_step(self, feedback, dt=1):
        e = self.ref - feedback
        if self.is_discrete:
            dt = self.sampling_time
        self.A += dt *  (e + self.last_err)/2
        P = self.weights[0] * e 
        I = self.weights[1] * self.A
        D = self.weights[2] * (e- self.last_err) / dt

        u = P + I + D
        print(u)
        self.last_err = u
        self.calculate_steering_and_publish(u)

   
    def calculate_steering_and_publish(self, offset):
        self.command = AckermannDriveStamped()
        self.command.drive.speed = self.speed


        if offset > 5.0 or offset < -5.0 or offset == 0.0:
            angle = 0.0
        elif offset > 0:
            angle = self.map_value(offset,0,5,0.0,0.34)
        elif offset < 0:
            angle = self.map_value(offset,0,-5,0.0,-0.34)

        self.command.drive.steering_angle = angle

        #self.pub_command.publish(command)

    def map_value(self, x, in_min, in_max, out_min, out_max):
        return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

if __name__ == "__main__":

    sc = SteeringCalculate(0,[0.6,0.01,0.01])
    
    try:
        #rospy.spin()
        while not rospy.is_shutdown():
            sc.pub_command.publish(sc.command)
            rospy.sleep(0.1)

    except KeyboardInterrupt:
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        sc.pub_command.publish(msg)
        print("Shutting down")
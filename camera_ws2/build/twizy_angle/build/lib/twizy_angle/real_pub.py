import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import NavSatFix
import numpy as np
import utm

class GNSSAnglePublisher(Node):
    def __init__(self):
        super().__init__('angle_real_publisher_gnss')
        self.publisher_ = self.create_publisher(Float32, 'angle_real', 10)

        self.subscription_twizzy = self.create_subscription(NavSatFix, 'gnss_twizzy', self.callback_twizzy, 10)
        self.subscription_trailer = self.create_subscription(NavSatFix, 'gnss_trailer', self.callback_trailer, 10)

        self.pos_twizzy = None
        self.pos_trailer = None

    def callback_twizzy(self, msg):
        self.pos_twizzy = utm.from_latlon(msg.latitude, msg.longitude)[:2]
        self.publish_angle()

    def callback_trailer(self, msg):
        self.pos_trailer = utm.from_latlon(msg.latitude, msg.longitude)[:2]
        self.publish_angle()

    def publish_angle(self):
        if self.pos_twizzy is None or self.pos_trailer is None:
            return

        dx = self.pos_trailer[0] - self.pos_twizzy[0]
        dy = self.pos_trailer[1] - self.pos_twizzy[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        angle_msg = Float32()
        angle_msg.data = angle_deg % 360  # en [0, 360)
        self.publisher_.publish(angle_msg)
        self.get_logger().info(f'Ángulo real publicado: {angle_msg.data:.2f}°')

def main(args=None):
    rclpy.init(args=args)
    node = GNSSAnglePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

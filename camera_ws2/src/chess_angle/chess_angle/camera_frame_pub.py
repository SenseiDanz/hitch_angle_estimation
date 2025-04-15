#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class CameraFramePublisher(Node):
    def __init__(self):
        super().__init__('camera_frame_publisher') 
        self.publisher_ = self.create_publisher(Image, 'camera_frame', 10)
        self.timer = self.create_timer(0.0333, self.publish_frame)  # 30 Hz
        self.bridge = CvBridge()

        self.serial_number = "046d_Logitech_BRIO_745683D1"
        self.cap = self.get_camera_by_serial(self.serial_number)
        #self.cap = cv2.VideoCapture(0)

        if not self.cap:
            self.get_logger().error("Error al abrir la cámara.")
            exit()
        
        # Establecer la resolución manualmente después de abrir la cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_camera_by_serial(self, serial):
        video_devices = [f'/dev/video{i}' for i in range(0, 8)]
        for dev in video_devices:
            cmd = f'udevadm info --query=all --name={dev}'
            udev_info = os.popen(cmd).read()
            if f'E: ID_SERIAL={serial}' in udev_info:
                cap = cv2.VideoCapture(dev)
                if cap.isOpened():
                    self.get_logger().info(f"Cámara encontrada: {dev}")
                    return cap
        return None

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("No se pudo capturar el frame.")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        self.publisher_.publish(msg)
        self.get_logger().info("Frame publicado")

    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    frame_publisher = CameraFramePublisher()
    try:
        rclpy.spin(frame_publisher)
    except KeyboardInterrupt:
        pass
    frame_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

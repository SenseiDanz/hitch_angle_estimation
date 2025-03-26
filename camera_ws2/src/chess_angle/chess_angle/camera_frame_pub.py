import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import cv2 
import os

class CameraFramePublisher(Node): 
    def init(self): 
        super().init('camera_frame_publisher') 
        self.publisher_ = self.create_publisher(Image, 'camera_frame', 10) 
        self.timer = self.create_timer(0.1, self.publish_frame) # Publica a 10 Hz self.bridge = CvBridge()
            # Número de serie de la cámara 
    self.serial_number = "046d_Logitech_BRIO_745683D1"
    self.cap = self.get_camera_by_serial(self.serial_number)
    if not self.cap:
        self.get_logger().error("Error al abrir la cámara.")
        exit()

    # Configurar resolución 
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def get_camera_by_serial(self, serial):
        video_devices = [f'/dev/video{i}' for i in range(0, 8)]
        for dev in video_devices:
            cmd = f'udevadm info --query=all --name={dev}'
            udev_info = os.popen(cmd).read()
            if f'E: ID_SERIAL={serial}' in udev_info:
                cap = cv2.VideoCapture(dev)
                if cap.isOpened():
                    self.get_logger().info(f"Cámara encontrada en {dev} con el número de serie {serial}.")
                    return cap
                else:
                    self.get_logger().error(f"No se pudo abrir el dispositivo {dev}.")
        self.get_logger().error(f"No se encontró la cámara con el número de serie {serial}.")
        return None

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("No se pudo capturar el frame de la cámara.")
            return

        # Convertir el frame de OpenCV a ROS Image
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera_frame"
        self.publisher_.publish(img_msg)
        self.get_logger().info("Publicando frame de la cámara.")

    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()

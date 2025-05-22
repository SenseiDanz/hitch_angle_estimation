#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import os
import time

class CameraFramePublisher(Node):
    def __init__(self):
        super().__init__('camera_frame_publisher') 
        self.publisher_ = self.create_publisher(Image, 'camera_frame', 10)
        self.status_pattern_pub = self.create_publisher(Bool, 'pattern_found', 10)
        self.timer = self.create_timer(0.0333, self.publish_frame)  # 30 Hz
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error("Error al abrir la cámara.")
            exit()
        
        # Establecer la resolución manualmente después de abrir la cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Configuración de exposición
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.25 para MANUAL, 0.75 para AUTO 
        self.MIN_EXPOSURE = -11.0
        self.MAX_EXPOSURE = -2.0
        self.EXPOSURE_SWITCH_THRESHOLD_LOW = -7.0
        self.EXPOSURE_SWITCH_THRESHOLD_HIGH = -5.0

        # Patrón de ajedrez
        self.CHECKERBOARD = (8, 11)

        # Control de intentos fallidos
        self.pattern_not_found_count = 0
        self.PATTERN_NOT_FOUND_LIMIT = 15

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pattern_found, _ = cv2.findChessboardCorners(gray, self.CHECKERBOARD, None)
        
        # Publicar estado del patrón
        pattern_msg = Bool()
        pattern_msg.data = bool(pattern_found)
        self.pattern_status_pub.publish(pattern_msg)
        self.get_logger().info(f"pattern_found = {pattern_found}")

        if not pattern_found:
            self.pattern_not_found_count += 1

            # Si es la primera vez que no se detecta el patrón, esperar un poco antes de actuar
            if self.pattern_not_found_count == 1:
                self.get_logger().info("Primera vez sin detección. Esperando antes de ajustar exposición.")
                time.sleep(1)  # espera 1s 
                return

            if self.pattern_not_found_count >= self.PATTERN_NOT_FOUND_LIMIT:
                self.get_logger().warn("Patrón probablemente no presente en la escena. Deteniendo ajustes.")
            else:
                exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) #Manual

                if exposure <= self.EXPOSURE_SWITCH_THRESHOLD_LOW and exposure > self.MIN_EXPOSURE:
                    new_exposure = exposure - 1.0
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                    self.get_logger().warn(f"Patrón no detectado. Bajando exposición a {new_exposure:.1f}")

                elif exposure >= self.EXPOSURE_SWITCH_THRESHOLD_HIGH and exposure < self.MAX_EXPOSURE:
                    new_exposure = exposure + 1.0
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                    self.get_logger().warn(f"Patrón no detectado. Subiendo exposición a {new_exposure:.1f}")

                elif exposure == -6.0:
                    new_exposure = exposure - 1.0
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                    self.get_logger().warn(f"Patrón no detectado en -6.0. Probando bajar exposición a {new_exposure:.1f}")

                elif exposure == self.MIN_EXPOSURE:
                    #new_exposure = exposure + 1.0
                    #self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    self.get_logger().warn(f"En límite inferior {exposure:.1f} . Aplicando CLAHE")

                elif exposure == self.MAX_EXPOSURE:
                    #new_exposure = exposure - 1.0
                    #self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    self.get_logger().warn(f"En límite superior {exposure:.1f} . Aplicando CLAHE.")

        else:
            self.pattern_not_found_count = 0  # Reinicia si se encuentra el patrón

        msg = self.bridge.cv2_to_imgmsg(gray, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        self.publisher_.publish(msg)
        self.get_logger().info("Frame publicado")


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

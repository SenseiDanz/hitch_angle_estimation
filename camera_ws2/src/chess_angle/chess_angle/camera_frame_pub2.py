#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class CameraFramePublisher2(Node):
    def __init__(self):
        super().__init__('camera_frame_publisher2') 
        self.publisher_ = self.create_publisher(Image, 'camera_frame2', 10)
        self.timer = self.create_timer(0.0333, self.publish_frame)  # 30 Hz
        self.bridge = CvBridge()

        # Número de serie de la cámara derecha
        self.serial_number =  "046d_Logitech_BRIO_F8C07419"

        # Intenta encontrar la cámara por número de serie
        self.cap = self.get_camera_by_serial(self.serial_number)

        if not self.cap.isOpened():
            self.get_logger().error("Error al abrir la cámara 2.")
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
                    self.get_logger().info(f"Cámara 2 encontrada: {dev}")
                    return cap
        return None

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("No se pudo capturar el frame.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pattern_found, _ = cv2.findChessboardCorners(gray, self.CHECKERBOARD, None)

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
                    new_exposure = exposure + 1.0
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                    self.get_logger().warn(f"En límite inferior (-11.0). Subiendo exposición a {new_exposure:.1f} para intentar recuperar patrón.")

                elif exposure == self.MAX_EXPOSURE:
                    new_exposure = exposure - 1.0
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                    self.get_logger().warn(f"En límite superior (-2.0). Bajando exposición a {new_exposure:.1f} para intentar recuperar patrón.")

        else:
            self.pattern_not_found_count = 0  # Reinicia si se encuentra el patrón

        msg = self.bridge.cv2_to_imgmsg(gray, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame2"
        self.publisher_.publish(msg)
        self.get_logger().info("Frame de la camara 2 publicado")


def main(args=None):
    rclpy.init(args=args)
    frame_publisher2 = CameraFramePublisher2()
    try:
        rclpy.spin(frame_publisher2)
    except KeyboardInterrupt:
        pass
    frame_publisher2.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

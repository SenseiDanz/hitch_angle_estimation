import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os

class AnglePublisher(Node):
    def __init__(self):
        super().__init__('angle_publisher')
        self.publisher_ = self.create_publisher(Float32, 'angle_topic', 10)
        self.subscription = self.create_subscription(Image, 'camera_frame', self.image_callback, 10)
        self.bridge = CvBridge()
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.angle = 0.0  
        self.calibrated = False

        self.width = 960
        self.height = 540
        self.dim = (self.width, self.height)

        # Calibrar cámara o cargar parámetros previos
        self.calibration_choice()

    def calibration_choice(self):
        print("\nSelecciona el patrón de ajedrez:")
        print("1. Patrón (5, 7) con square_size = 2.5")
        print("2. Patrón (8, 11) con square_size = 6.0")
        opcion = input("Ingresa 1 o 2 según el patrón que estés usando: ").strip()

        if opcion == '1':
            self.CHECKERBOARD = (5, 7)
            self.square_size = 2.5
            self.calibration_file = "calibration_5x7.yml"
        elif opcion == '2':
            self.CHECKERBOARD = (8, 11)
            self.square_size = 6.0
            self.calibration_file = "calibration_8x11.yml"
        else:
            print("Opción inválida. Se usará el patrón por defecto (5, 7).")
            self.CHECKERBOARD = (5, 7)
            self.square_size = 2.5
            self.calibration_file = "calibration_5x7.yml"

        # Regenerar objp según el patrón elegido
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0],
                                    0:self.CHECKERBOARD[1]].T.reshape(-1, 2) * self.square_size

        user_input = input("¿Deseas calibrar la cámara? (s/n): ").strip().lower()
        if user_input == 's':
            self.get_logger().info(f"Iniciando calibración con patrón {self.CHECKERBOARD} y square_size = {self.square_size}")
            self.calibrate_camera()
        else:
            self.load_calibration_params()

    def calibrate_camera(self):
        self.get_logger().info("Capturando 60 imágenes para calibración...")
        objpoints = []
        imgpoints = []
        cap = cv2.VideoCapture(0)
        # Establecer la resolución manualmente después de abrir la cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            self.get_logger().error("No se pudo abrir la cámara para calibrar.")
            return
        else:
            self.get_logger().info("Cámara abierta")

        count = 0
        last_capture_time = time.time()
        min_interval = 0.5  # segundos entre capturas
        while count < 60:
            ret, frame = cap.read()
            if not ret:
                self.get_logger().error("No se pudo capturar el frame.")
                continue

            current_time = time.time()
            if current_time - last_capture_time < min_interval:
                continue  # esperar un poco antes de procesar otro frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)
                #cv2.drawChessboardCorners(frame, self.CHECKERBOARD, corners2, ret)
                count += 1
                last_capture_time = current_time  # reinicia el reloj
                self.get_logger().info(f"Imagen de calibración capturada: {count}/60")
                
            #frame_resized = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
            #cv2.imshow("Calibración - Cámara", frame_resized)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break

        cap.release()
        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.calibrated = True
        self.get_logger().info("Calibración completada.")
        self.save_calibration_params()

    def save_calibration_params(self):
        fs = cv2.FileStorage(self.calibration_file, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", self.mtx)
        fs.write("dist_coeffs", self.dist)
        fs.write("rvecs", np.array(self.rvecs))
        fs.write("tvecs", np.array(self.tvecs))
        fs.release()
        self.get_logger().info(f"Parámetros guardados en {self.calibration_file}")

    def load_calibration_params(self):
        fs = cv2.FileStorage(self.calibration_file, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            self.get_logger().error(f"No se pudo abrir el archivo {self.calibration_file}")
            return

        self.mtx = fs.getNode("camera_matrix").mat()
        self.dist = fs.getNode("dist_coeffs").mat()
        fs.release()
        self.calibrated = True
        self.get_logger().info(f"Parámetros de calibración cargados desde {self.calibration_file}")

    def image_callback(self, msg):
        start_time = time.time()

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        angle = self.calculate_angle(frame)

        # Luego resize solo para mostrar
        #frame_resized = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)

        angle_msg = Float32()
        angle_msg.data = angle
        self.publisher_.publish(angle_msg)
        self.get_logger().info(f"Ángulo publicado: {angle:.2f} grados")

        #cv2.putText(frame_resized, f'Angulo: {angle:.2f} grados', (10, 30),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        #cv2.imshow("Cámara - Ángulo estimado", frame_resized)

        end_time = time.time()
        elapsed = end_time - start_time
        self.get_logger().info(f"Tiempo de iteración: {elapsed:.4f} segundos")

        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #cv2.destroyAllWindows()

    def calculate_angle(self, frame):
        if not self.calibrated:
            return self.angle

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            success, rvec, tvec = cv2.solvePnP(
                self.objp, corners, self.mtx, self.dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            rvec, tvec = cv2.solvePnPRefineLM(
                self.objp, corners, self.mtx, self.dist, rvec, tvec
            )

            R, _ = cv2.Rodrigues(rvec)

        
            yaw = np.arctan2(R[0, 2], R[2, 2])  # eje y de la cámara

            angle_deg = np.degrees(yaw)
            if angle_deg < 0:
                angle_deg = 360 + angle_deg  # pasa de [-180, 0) a [180, 360)

            # Suavizado circular
            alpha = 0.9
            prev_rad = np.radians(self.angle)
            new_rad = np.radians(angle_deg)

            x = alpha * np.cos(prev_rad) + (1 - alpha) * np.cos(new_rad)
            y = alpha * np.sin(prev_rad) + (1 - alpha) * np.sin(new_rad)

            smoothed_angle = np.degrees(np.arctan2(y, x))
            if smoothed_angle < 0:
                smoothed_angle += 360

            self.angle = smoothed_angle

        else:
            self.get_logger().warn("No se detectó el patrón de ajedrez. Manteniendo ángulo anterior.")

        return self.angle


def main(args=None):
    rclpy.init(args=args)
    angle_publisher = AnglePublisher()
    rclpy.spin(angle_publisher)
    angle_publisher.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


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

        self.angle = 0.0  # Ángulo estimado actual
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
            print("Opción inválida. Se usará el patrón por defecto (8, 11).")
            self.CHECKERBOARD = (8, 11)
            self.square_size = 6.0
            self.calibration_file = "calibration_8x11.yml"

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

        # Configuración de exposición
        MIN_EXPOSURE = -11.0
        MAX_EXPOSURE = -2.0
        EXPOSURE_SWITCH_THRESHOLD_LOW = -7.0
        EXPOSURE_SWITCH_THRESHOLD_HIGH = -5.0
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # 0.25 para MANUAL, 0.75 para AUTO

        # Control de intentos fallidos
        pattern_not_found_count = 0
        PATTERN_NOT_FOUND_LIMIT = 15
        
        while count < 60:
            ret, frame = cap.read()
            if not ret:
                self.get_logger().error("No se pudo capturar el frame.")
                continue

            current_time = time.time()
            if current_time - last_capture_time < min_interval:
                continue  # esperar un poco antes de procesar otro frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pattern_found, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)
            if pattern_found:
                objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)
                #cv2.drawChessboardCorners(frame, self.CHECKERBOARD, corners2, ret)
                count += 1
                last_capture_time = current_time
                pattern_not_found_count = 0  # Reinicia si se encuentra el patrón
                self.get_logger().info(f"Imagen de calibración capturada: {count}/60")
            else:
                pattern_not_found_count += 1

                # Si es la primera vez que no se detecta el patrón, esperar un poco antes de actuar
                if pattern_not_found_count == 1:
                    self.get_logger().info("Primera vez sin detección. Esperando antes de ajustar exposición.")
                    time.sleep(1)  # espera 1s 
                    return

                if pattern_not_found_count >= PATTERN_NOT_FOUND_LIMIT:
                    self.get_logger().warn("Patrón probablemente no presente en la escena. Deteniendo ajustes.")
                else:
                    exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) #Manual

                    if exposure <= EXPOSURE_SWITCH_THRESHOLD_LOW and exposure > MIN_EXPOSURE:
                        new_exposure = exposure - 1.0
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                        self.get_logger().warn(f"Patrón no detectado. Bajando exposición a {new_exposure:.1f}")

                    elif exposure >= EXPOSURE_SWITCH_THRESHOLD_HIGH and exposure < MAX_EXPOSURE:
                        new_exposure = exposure + 1.0
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                        self.get_logger().warn(f"Patrón no detectado. Subiendo exposición a {new_exposure:.1f}")

                    elif exposure == -6.0:
                        new_exposure = exposure - 1.0
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                        self.get_logger().warn(f"Patrón no detectado en -6.0. Probando bajar exposición a {new_exposure:.1f}")

                    elif exposure == MIN_EXPOSURE:
                        #new_exposure = exposure + 1.0
                        #self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        gray = clahe.apply(gray)
                        self.get_logger().warn(f"En límite inferior {exposure:.1f} . Aplicando CLAHE")

                    elif exposure == MAX_EXPOSURE:
                        #new_exposure = exposure - 1.0
                        #self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        gray = clahe.apply(gray)
                        self.get_logger().warn(f"En límite superior {exposure:.1f} . Aplicando CLAHE.")
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

        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8') # frame convertido ya a escala de grises
        angle = self.calculate_angle(gray)

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

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            frame,
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

            # Aquí extraemos el yaw
            yaw = np.arctan2(R[0, 2], R[2, 2])  # eje y de la cámara

            angle_deg = np.degrees(yaw)           
            self.angle = angle_deg
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
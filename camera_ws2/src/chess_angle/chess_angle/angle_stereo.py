import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os

class StereoAnglePublisher(Node):
    def __init__(self):
        super().__init__('stereo_angle_publisher')
        self.publisher_ = self.create_publisher(Float32, 'angle_topic', 10)
        self.subscription = self.create_subscription(Image, 'camera_frame1', self.image_callback1, 10)
        self.subscription2 = self.create_subscription(Image, 'camera_frame2', self.image_callback2, 10)
        self.bridge = CvBridge()

        self.angle = 0.0
        self.frame1 = None
        self.frame2 = None

        self.calibration_choice()

    def get_camera_by_serial(self, serial):
        video_devices = [f'/dev/video{i}' for i in range(0, 8)]
        for dev in video_devices:
            cmd = f'udevadm info --query=all --name={dev}'
            udev_info = os.popen(cmd).read()
            if f'E: ID_SERIAL={serial}' in udev_info:
                cap = cv2.VideoCapture(dev)
                if cap.isOpened():
                    self.get_logger().info(f"Cámara 1 encontrada: {dev}")
                    return cap
        return None

    def calibration_choice(self):
        # Selección de patrón
        print("\nSelecciona el patrón de ajedrez:")
        print("1. Patrón (5, 7) con square_size = 2.5")
        print("2. Patrón (8, 11) con square_size = 6.0")
        opcion = input("Ingresa 1 o 2 según el patrón que estés usando: ").strip()

        if opcion == '1':
            self.CHECKERBOARD = (5, 7)
            self.square_size = 2.5
        elif opcion == '2':
            self.CHECKERBOARD = (8, 11)
            self.square_size = 6.0
        else:
            print("Opción inválida. Se usará el patrón por defecto (8, 11).")
            self.CHECKERBOARD = (8, 11)
            self.square_size = 6.0

        self.objp = np.zeros((np.prod(self.CHECKERBOARD), 3), np.float32)
        self.objp[:, :2] = np.indices(self.CHECKERBOARD).T.reshape(-1, 2) * self.square_size

        # Selección de calibración
        user_input = input("¿Deseas calibrar la cámara? (s/n): ").strip().lower()
        if user_input == 's':
            # Número de serie
            self.serial_numberL = "046d_Logitech_BRIO_745683D1"
            self.serial_numberR =  "046d_Logitech_BRIO_F8C07419"
        # Intenta encontrar la cámara por número de serie
            self.capL = self.get_camera_by_serial(self.serial_numberL)
            self.capR = self.get_camera_by_serial(self.serial_numberR)
            self.get_logger().info(f"Iniciando calibración de la cámara 1 con patrón {self.CHECKERBOARD} y square_size = {self.square_size}")
            self.calibrate_camera(capL,1)
            self.get_logger().info(f"Iniciando calibración de la cámara 2 ")
            self.calibrate_camera(capR,2)
        else:
            self.load_stereo_calibration()
        

    def calibrate_camera(self,cap,cam_num):
        self.get_logger().info("Capturando 60 imágenes para calibración...")
        objpoints = []
        imgpoints = []

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
                        new_exposure = exposure + 1.0
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                        self.get_logger().warn(f"En límite inferior (-11.0). Subiendo exposición a {new_exposure:.1f} para intentar recuperar patrón.")

                    elif exposure == MAX_EXPOSURE:
                        new_exposure = exposure - 1.0
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                        self.get_logger().warn(f"En límite superior (-2.0). Bajando exposición a {new_exposure:.1f} para intentar recuperar patrón.")
                
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
        self.save_calibration_params(cam_num)

    def save_calibration_params(self,num):
        fs = cv2.FileStorage(self.calibration_file, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", self.mtx)
        fs.write("dist_coeffs", self.dist)
        fs.write("rvecs", np.array(self.rvecs))
        fs.write("tvecs", np.array(self.tvecs))
        fs.release()
        self.get_logger().info(f"Parámetros guardados en {self.calibration_file}")

    def load_stereo_calibration(self):
        fs = cv2.FileStorage('stereo_calibration.yml', cv2.FILE_STORAGE_READ)
        self.K1 = fs.getNode('K1').mat()
        self.D1 = fs.getNode('D1').mat()
        self.K2 = fs.getNode('K2').mat()
        self.D2 = fs.getNode('D2').mat()
        self.P1 = fs.getNode('P1').mat()
        self.P2 = fs.getNode('P2').mat()
        fs.release()

    def image_callback1(self, msg):
        self.frame1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.try_process()

    def image_callback2(self, msg):
        self.frame2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.try_process()

    def try_process(self):
        if self.frame1 is None or self.frame2 is None:
            return

        gray1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2GRAY)

        found1, corners1 = cv2.findChessboardCorners(gray1, self.CHECKERBOARD, None)
        found2, corners2 = cv2.findChessboardCorners(gray2, self.CHECKERBOARD, None)

        if found1 and found2:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), term)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), term)

            undist1 = cv2.undistortPoints(corners1, self.K1, self.D1, P=self.P1)
            undist2 = cv2.undistortPoints(corners2, self.K2, self.D2, P=self.P2)

            points_4d = cv2.triangulatePoints(self.P1, self.P2, undist1, undist2)
            points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
            points_3d = points_3d.reshape(-1, 3)

            _, _, vt = np.linalg.svd(points_3d - np.mean(points_3d, axis=0))
            normal = vt[-1]

            vehicle_forward = np.array([0, 0, 1])
            angle_rad = np.arctan2(np.cross(vehicle_forward, normal)[1], np.dot(vehicle_forward, normal))
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360

            self.angle = angle_deg
            angle_msg = Float32()
            angle_msg.data = angle_deg
            self.publisher_.publish(angle_msg)
            self.get_logger().info(f"Ángulo publicado: {angle_deg:.2f} grados")
        else:
            self.get_logger().warn("No se detectó el patrón en ambas cámaras.")


def main(args=None):
    rclpy.init(args=args)
    node = StereoAnglePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

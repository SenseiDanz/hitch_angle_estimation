import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2
import numpy as np
import os
import time

class AnglePublisher(Node):
    def __init__(self):
        super().__init__('angle_publisher')
        self.publisher_ = self.create_publisher(Float32, 'angle_topic', 10)
        self.timer = self.create_timer(0.01, self.publish_angle)  # Publica cada 0.10s (100 Hz)
        
        # Número de serie de la cámara
        self.serial_number = "046d_Logitech_BRIO_745683D1"
        
        # Intenta encontrar la cámara por número de serie
        self.cap = self.get_camera_by_serial(self.serial_number)
        
        # Establecer la resolución manualmente después de abrir la cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Ancho 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Alto 
        
        if not self.cap:
            self.get_logger().error("Error al abrir la cámara.")
            exit()

        # Definición de las dimensiones del tablero de ajedrez
        self.CHECKERBOARD = (5, 7)  # Cambié el número de esquinas a 5 y 7
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Vectores para almacenar puntos 3D y 2D para cada imagen del tablero
        self.objpoints = []
        self.imgpoints = []
        
        # Definir las coordenadas del mundo para los puntos 3D
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2) * 2.5
        
        self.angle = 0.0
      
        # Para ajustar tamaño de las ventanas
        self.width = 960
        self.height = 540
        self.dim = (self.width, self.height)
        
        # Llamada a la función de calibración o carga de parámetros
        self.calibration_choice()

    def get_camera_by_serial(self, serial):
        # Intentamos abrir varios dispositivos de video
        video_devices = [f'/dev/video{i}' for i in range(0, 8)]  # desde /dev/video0 hasta /dev/video7
        self.get_logger().info(f"Dispositivos encontrados: {video_devices}")
        
        for dev in video_devices:
            device_path = dev
            # Usar udevadm para obtener la información del dispositivo y comprobar si tiene el número de serie
            cmd = f'udevadm info --query=all --name={device_path}'
            udev_info = os.popen(cmd).read()
            self.get_logger().info(f"Información del dispositivo {dev}: {udev_info}")
            if f'E: ID_SERIAL={serial}' in udev_info:
                self.get_logger().info(f"Cámara encontrada con el número de serie {serial}.")
                # Crear un VideoCapture con la ruta del dispositivo encontrado
                cap = cv2.VideoCapture(device_path)
                if cap.isOpened():
                    return cap
                else:
                    self.get_logger().error(f"No se pudo abrir el dispositivo {device_path}.")
        
        self.get_logger().error(f"No se encontró la cámara con el número de serie {serial}.")
        return None

    def calibration_choice(self):
        # Opción para calibrar o cargar parámetros guardados
        user_input = input("¿Deseas calibrar la cámara? (s/n): ").strip().lower()
        if user_input == 's':
            self.calibrate_camera()
        else:
            self.load_calibration_params()

    def calibrate_camera(self):
        # Capturar frames suficientes para la calibración
        frame_count = 0
        while frame_count < 60:  # Capturamos 60 frames
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("No se pudo capturar el frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                frame_count += 1
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)
                cv2.drawChessboardCorners(frame, self.CHECKERBOARD, corners2, ret)
            
            frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("Calibración de Cámara", frame)
            cv2.waitKey(1)

        # Realizar la calibración de la cámara
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.calibrated = True
        self.get_logger().info("Calibración de la cámara completada.")
        cv2.destroyWindow("Calibración de Cámara")
        
        # Guardar los parámetros de calibración
        self.save_calibration_params()

    def save_calibration_params(self):
        fs = cv2.FileStorage("calibration_params.yml", cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", self.mtx)
        fs.write("dist_coeffs", self.dist)
        fs.write("rvecs", np.array(self.rvecs))
        fs.write("tvecs", np.array(self.tvecs))
        fs.release()

    def load_calibration_params(self):
        fs = cv2.FileStorage("calibration_params.yml", cv2.FILE_STORAGE_READ)
        self.mtx = fs.getNode("camera_matrix").mat()
        self.dist = fs.getNode("dist_coeffs").mat()
        self.rvecs = fs.getNode("rvecs").mat()
        self.tvecs = fs.getNode("tvecs").mat()
        fs.release()
        self.calibrated = True
        self.get_logger().info("Parámetros de calibración cargados con éxito.")

    def calculate_angle(self, frame):
        if self.calibrated:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                retval, rvec, tvec = cv2.solvePnP(self.objp, corners, self.mtx, self.dist)

                if retval:
                    R, _ = cv2.Rodrigues(rvec)
                    normal = np.array([0, 0, 1])
                    cam_z = R[:, 2]
                    cos_angle = np.dot(normal, cam_z) / (np.linalg.norm(normal) * np.linalg.norm(cam_z))
                    angle = np.arccos(cos_angle)
                    self.angle = np.degrees(angle)
                    cv2.drawChessboardCorners(frame, self.CHECKERBOARD, corners, ret)

        return frame

    def publish_angle(self):
        angle_msg = Float32()
        angle_msg.data = self.angle
        self.publisher_.publish(angle_msg)
        self.get_logger().info(f'Publicando ángulo: {self.angle:.2f} grados')

    def run(self):
        while rclpy.ok():
            self.get_logger().info("Intentando capturar frame...")
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("No se pudo capturar el frame.")
                break
                
            # Para observar cuanto tarda cada iteración 
            start_time =time.time()
            # Calcular el ángulo con el frame actual
            frame = self.calculate_angle(frame)
            frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
            # Mostrar el frame con el patrón de ajedrez y el ángulo calculado
            cv2.putText(frame, f'Angulo: {self.angle:.2f} grados', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Cámara. Presiona 'q' para terminar le programa", frame)

            self.publish_angle()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.get_logger().info(f'Tiempo de iteración: {elapsed_time:.4f} segundos')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    angle_publisher = AnglePublisher()
    angle_publisher.run()
    angle_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


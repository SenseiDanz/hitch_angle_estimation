import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import imutils
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import os
import math

# Diccionario de segmentos para identificar los dígitos del goniometro
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 0, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (1, 0, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 1, 1, 0, 1, 0): 4,
    (0, 1, 0, 1, 0, 0, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 0, 0, 1, 0, 1, 1): 5,
    (1, 0, 0, 0, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 1, 0, 1, 1, 0, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

class AngleRealPublisher(Node):
    def __init__(self):
        super().__init__('angle_real_publisher')
        self.publisher_ = self.create_publisher(Float32, 'angle_real', 10)
        self.timer = self.create_timer(0.01, self.publish_angle)  # Publica cada 0.01s (100 hz)
        
        # Número de serie de la cámara
        self.serial_number = "046d_Logitech_BRIO_F8C07419"
        
        # Configuración para la cámara del goniómetro
        self.cap = self.get_camera_by_serial(self.serial_number)  # Cámara 2 para el goniómetro
        
        # Establecer la resolución manualmente después de abrir la cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Ancho 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Alto 
        
        if not self.cap.isOpened():
            self.get_logger().error("Error al abrir la cámara.")
            exit()

        # Control de ventana (inicialmente a color)
        self.show_color = True

    def get_camera_by_serial(self, serial):
        # Intentamos abrir varios dispositivos de video
        video_devices = [f'/dev/video{i}' for i in range(0, 8)]  # desde /dev/video0 hasta /dev/video7
        
        for dev in video_devices:
            device_path = dev
            # Usar udevadm para obtener la información del dispositivo y comprobar si tiene el número de serie
            cmd = f'udevadm info --query=all --name={device_path}'
            udev_info = os.popen(cmd).read()
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

    def publish_angle(self):
        # Procesar la imagen de la cámara y obtener el ángulo real
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("No se pudo capturar el frame de la cámara del goniometro.")
            return

        # Preprocesar la imagen
        image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        image = imutils.resize(image, height=700)
        image = image[275:435,125:370]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Color", image)  # Mostrar imagen a color
        cv2.waitKey(1)

        # Preprocesamiento: CLAHE y filtro de suavizado
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)
        blurred = cv2.GaussianBlur(cl1, (5, 5), 1)
        edged = cv2.Canny(blurred, 200, 255, 255)

        # Encontrar contornos
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        displayCnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                displayCnt = approx
                break

        if displayCnt is not None:
            warped = four_point_transform(cl1, displayCnt.reshape(4, 2))
            warped = warped[18:90, 0:180]
            warped = cv2.copyMakeBorder(warped, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(160, 160, 160))
            cv2.imshow("Warped", warped)  # Mostrar imagen transformada

            output = four_point_transform(image, displayCnt.reshape(4, 2))
            output = output[18:150, 0:190]
            
            thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,19,30)
            cv2.imshow("Threshold", thresh)  # Mostrar imagen umbralizada

            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 10))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 4))
            kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11))
            kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

            morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
            cv2.imshow("Morph1", morph1)  # Mostrar morph1
            morph2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2)
            cv2.imshow("Morph2", morph2)  # Mostrar morph2
            suma = cv2.add(morph1,morph2)
            cv2.imshow("suma", suma)
            dilat = cv2.dilate(suma,kernel4,1)
            cv2.imshow("dilatar", dilat)
            morph3 = cv2.morphologyEx(suma, cv2.MORPH_CLOSE, kernel3)
            cv2.imshow("Morph3", morph3)  # Mostrar morph3
            
            thresh = morph3
            cv2.imwrite('thresh.png',thresh)

            # Detectar los dígitos
            digitCnts = []
            cnts2 = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts2 = imutils.grab_contours(cnts2)

            if len(cnts2) == 0:
                self.get_logger().error("No se detectaron contornos para los dígitos.")
                return

            for c in cnts2:
                (x, y, w, h) = cv2.boundingRect(c)
                if w >= 4 and h >= 45 and h <= 60:
                    digitCnts.append(c)

            if len(digitCnts) == 0:
                self.get_logger().error("No se encontraron contornos válidos para los dígitos.")
                return

            digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
            digits = []

            for c in digitCnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if w <= 15: # para diferenciar el roi del número 1
                   w2=28
                   roi = thresh[y:y + h, x - (w2-w):x + w]
                   w=w2
                else:               
                   roi = thresh[y:y + h, x:x + w]
               
                (roiH, roiW) = roi.shape
                (dW, dH) = (int(roiW * 0.25), int(roiH * 0.12))

                segments = [
                    ((0, 0), (w, dH)),
                    ((0, 0), (dW, h // 2)),
                    ((w - dW, 0), (w, h // 2)),
                    ((0, (h // 2) - dH), (w, (h // 2) + dH)),
                    ((0, h // 2), (dW, h)),
                    ((w - dW, h // 2), (w, h)),
                    ((0, h - dH), (w, h))
                ]
                on = [0] * len(segments)

                for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                    segROI = roi[yA:yB, xA:xB]
                    total = cv2.countNonZero(segROI) #número de pixeles blancos
                    area = (xB - xA) * (yB - yA)
                    if total / float(area) > 0.4:
                        on[i] = 1

                # Depurar el patrón
                try:
                   digit = DIGITS_LOOKUP[tuple(on)]
                   digits.append(str(digit))
                   cv2.rectangle(output, (x, y), (x - w, y + h), (0, 255, 0), 1)
                   cv2.putText(output, str(digit), (x -25, y + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                   cv2.imshow("Output", output)
                   cv2.waitKey(1)
                except KeyError:
                # Si ocurre un KeyError, muestra el patrón y continúa
                   self.get_logger().warn(f"Patrón no encontrado: {tuple(on)}")
                   digits.append('?')  # Asignar un valor que indique un patrón no reconocido
                   
            # Si no se detecta ningún dígito válido, no continuar con el cálculo
            if len(digits) < 3 or '?' in digits:
               self.get_logger().warn("No se detectó un ángulo válido. Continuando con el siguiente frame...")
               return
               
            # Convertir a float
            angle_str = ''.join(digits)
            angle_real = float(angle_str) / 100.0 # Ajusta la escala según sea necesario
            
            # Ajuste ángulo 360
            if angle_real < 180.0:
               angle_real = angle_real % 360.0
            else:
               angle_real = (360.0 - angle_real) % 360.0

            # Publicar el ángulo real 
            angle_real_msg = Float32()
            angle_real_msg.data = angle_real
            self.publisher_.publish(angle_real_msg)
            self.get_logger().info(f'Publicando ángulo real: {angle_real:.2f} grados')

    def run(self):
        while rclpy.ok():
            self.publish_angle()

def main(args=None):
    rclpy.init(args=args)
    angle_real_publisher = AngleRealPublisher()
    rclpy.spin(angle_real_publisher)
    angle_real_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



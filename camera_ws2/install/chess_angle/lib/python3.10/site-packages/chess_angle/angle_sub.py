import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time
import csv
import os
import numpy as np

class AngleSubscriber(Node):
    def __init__(self):
        super().__init__('error_calculator')
        self.subscription_angle_calculated = self.create_subscription(
            Float32, 'angle_topic', self.angle_calculated_callback, 10)
        self.subscription_angle_real = self.create_subscription(
            Float32, 'angle_real', self.angle_real_callback, 10)
            
        self.publisher_error = self.create_publisher(Float32, 'angle_error', 10)
        
        self.angle_calculated = None
        self.angle_real = None
        
        self.prev_time = None  # Para calcular la latencia
        self.start_time = None  # Para reiniciar el tiempo desde 0

        # Historial para almacenar ángulos y timestamps
        self.angle_real_history = []  # (timestamp, angle_real)
        
        # Suavizado: factor de suavizado para la variabilidad del ángulo real
        self.smoothing_factor = 0.1  

        # Función para generar un nombre único para el archivo CSV
        self.csv_filename = self.generar_nombre_archivo("angulos_timestamps.csv")
        
        # Crear el archivo CSV si no existe y agregar cabecera
        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'angle_real', 'angle_calculated', 'error'])

    def generar_nombre_archivo(self, nombre_archivo_base):
        filename = nombre_archivo_base
        contador = 1
        while os.path.exists(filename):
            filename = f"{nombre_archivo_base[:-4]}_{contador}.csv"
            contador += 1
        return filename
        
    def angle_calculated_callback(self, msg):
        self.angle_calculated = msg.data
        self.calculate_error()
    
    def angle_real_callback(self, msg):
        self.angle_real = msg.data
        self.calculate_error()

    def smooth_angle(self, new_angle):
        if len(self.angle_real_history) > 0:
            prev_angle = self.angle_real_history[-1][1]
            if abs(new_angle - prev_angle) > 5:  
                smoothed_angle = prev_angle + (new_angle - prev_angle) * self.smoothing_factor
            else:
                smoothed_angle = new_angle
        else:
            smoothed_angle = new_angle
        
        return smoothed_angle
    
    def calculate_error(self):
        current_time = self.get_clock().now()

        # Iniciar el tiempo desde cero en la primera recepción
        if self.start_time is None:
            self.start_time = current_time

        timestamp = (current_time - self.start_time).nanoseconds / 1e9  # En segundos

        # Calcular la latencia
        if self.prev_time is not None:
            latency = current_time - self.prev_time
            self.get_logger().info(f'Latencia: {latency.nanoseconds / 1e6:.2f} ms')

        # Suavizar el ángulo real
        if self.angle_real is not None:
            smoothed_angle = self.smooth_angle(self.angle_real)
            self.angle_real = smoothed_angle  
            self.angle_real_history.append((timestamp, self.angle_real))

        # Si ambos ángulos existen, calcular el error
        if self.angle_calculated is not None and self.angle_real is not None:
            self.get_logger().info(f'Ángulo estimado: {self.angle_calculated:.2f} grados')
            self.get_logger().info(f'Ángulo real (suavizado): {self.angle_real:.2f} grados')
            
            error = abs(self.angle_calculated - self.angle_real)
            error_msg = Float32()
            error_msg.data = error
            self.publisher_error.publish(error_msg)
            self.get_logger().info(f'Error: {error:.2f} grados')

            # Guardar en CSV
            with open(self.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, self.angle_real, self.angle_calculated, error])

        self.prev_time = current_time

def main(args=None):
    rclpy.init(args=args)
    angle_subscriber = AngleSubscriber()
    rclpy.spin(angle_subscriber)
    angle_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



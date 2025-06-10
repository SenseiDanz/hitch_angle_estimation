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
            Float32, '/twizy/gnss/angle_topic', self.angle_calculated_callback, 10)

        self.subscription_angle_hitch = self.create_subscription(
            Float32, '/twizy/gnss/hitch', self.angle_hitch_callback, 10) 

        self.subscription_angle_beta = self.create_subscription(
            Float32, '/twizy/gnss/beta', self.angle_beta_callback, 10)  
            
        self.publisher_error = self.create_publisher(Float32, 'angle_error', 10)
        
        self.angle_calculated = None
        self.angle_hitch = None
        self.angle_beta = None
        
        self.prev_time = None  # Para calcular la latencia
        self.start_time = None  # Para reiniciar el tiempo desde 0

        # Función para generar un nombre único para el archivo CSV
        self.csv_filename = self.generar_nombre_archivo("angulos_timestamps.csv")
        
        # Crear el archivo CSV si no existe y agregar cabecera
        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'angle_real_hitch', 'error_hitch',
                'angle_real_beta', 'error_beta', 'angle_calculated'
            ])

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
    
    def angle_hitch_callback(self, msg):
        # Convertir de radianes a grados
        self.angle_hitch = np.degrees(msg.data)
        self.calculate_error()

    def angle_beta_callback(self, msg):
        # Convertir de radianes a grados
        self.angle_beta = np.degrees(msg.data)
        self.calculate_error()

    
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

        # Si los ángulos existen, calcular el error
        if self.angle_calculated is not None:
            error_hitch = None
            error_beta = None

            if self.angle_hitch is not None:
                error_hitch = abs(self.angle_calculated - self.angle_hitch)
                self.get_logger().info(f'Ángulo real (hitch): {self.angle_hitch:.2f} grados')
                self.get_logger().info(f'Error con hitch: {error_hitch:.2f} grados')

            if self.angle_beta is not None:
                error_beta = abs(self.angle_calculated - self.angle_beta)
                self.get_logger().info(f'Ángulo real (beta): {self.angle_beta:.2f} grados')
                self.get_logger().info(f'Error con beta: {error_beta:.2f} grados')

            self.get_logger().info(f'Ángulo estimado: {self.angle_calculated:.2f} grados')

            # Guardar en CSV
            with open(self.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    f"{timestamp:.3f}",
                    f"{self.angle_hitch:.2f}" if self.angle_hitch is not None else '',
                    f"{error_hitch:.2f}" if error_hitch is not None else '',
                    f"{self.angle_beta:.2f}" if self.angle_beta is not None else '',
                    f"{error_beta:.2f}" if error_beta is not None else '',
                    f"{self.angle_calculated:.2f}"
                ])
        self.prev_time = current_time

def main(args=None):
    rclpy.init(args=args)
    angle_subscriber = AngleSubscriber()
    rclpy.spin(angle_subscriber)
    angle_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth_image', 10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.bridge = CvBridge()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Unable to open the camera.")
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to capture image.")
            return

        image_message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(image_message)
        self.get_logger().info('Publishing RGB image frame')

        height, width, _ = frame.shape
        fake_depth = np.full((height, width), 1000, dtype=np.uint16)

        depth_message = self.bridge.cv2_to_imgmsg(fake_depth, encoding='mono16')
        self.depth_pub.publish(depth_message)
        self.get_logger().info('Publishing fake depth image')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    camera_publisher = CameraPublisher()

    rclpy.spin(camera_publisher)

    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

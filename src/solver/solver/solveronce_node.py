from interface.msg import Point3D, Object
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
import yaml

class SolverOnceNode(Node):
    def __init__(self):
        super().__init__('solveronce_node')

        self.camera_matrix, self.dist_coeffs = self.load_camera_config('/home/zhujunheng/ros_yolo/src/solver/solver/config/camera.yaml')

        self.points_pub = self.create_publisher(Point3D, '/solveronce/result_point', 10)

        self.subscription = self.create_subscription(
            Object,
            '/decision/matched_object',
            self.listener_callback,
            10
        )

        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth_image',
            self.depth_callback,
            10
        )

        self.bridge = CvBridge()
        self.depth_image = None

    def depth_callback(self, depth_msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        self.get_logger().info('Received depth image')

    def listener_callback(self, obj):
        if self.depth_image is None:
            self.get_logger().warning("No depth image available yet.")
            return

        u = (obj.bbox[0] + obj.bbox[2]) / 2  
        v = (obj.bbox[1] + obj.bbox[3]) / 2 
        
        depth = self.get_depth_at(u, v)

        point_3d = self.pixel_to_3d(u, v, depth, self.camera_matrix)

        point_msg = Point3D(
            unique_id=0,
            id=obj.id,
            confidence=obj.confidence,
            x=point_3d[0],
            y=point_3d[1],
            z=point_3d[2]
        )

        self.points_pub.publish(point_msg)
        self.get_logger().info(f'Published 3D point: {point_3d}')

    def get_depth_at(self, u, v):
        return self.depth_image[int(v), int(u)]

    def pixel_to_3d(self, u, v, depth, camera_matrix):
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # 使用深度值计算3D坐标
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return np.array([x, y, z])

    def load_camera_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        camera_matrix = np.array(config['camera_matrix']['data']).reshape(3, 3)
        dist_coeffs = np.array(config['distortion_coefficients']['data'])
        return camera_matrix, dist_coeffs

def main(args=None):
    rclpy.init(args=args)
    solver_once_node = SolverOnceNode()

    rclpy.spin(solver_once_node)

    solver_once_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from interface.msg import CheckResult, Object
from .yolo import YOLO
import yaml

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        package_share_directory = '/home/zhujunheng/ros_yolo/src/inference/inference'
        config_file_path = package_share_directory + '/config/config.yaml'

        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_path = package_share_directory + '/model/' + config['model_name']
        self.labels = config['labels']
        self.model_h = config['model_h']
        self.model_w = config['model_w']

        self.subscription = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.listener_callback, 
            10
        )
        
        self.publisher_ = self.create_publisher(CheckResult, '/inference/output', 10)
        self.image_pub = self.create_publisher(Image, '/inference/image', 10)

        self.yolo = YOLO(
            model_path=self.model_path,
            labels=self.labels,
            model_h=self.model_h,
            model_w=self.model_w,
        )

        self.bridge = CvBridge()

    def listener_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        boxes, confs, ids = self.yolo.infer_img(cv_image)

        output_msg = CheckResult()
        for box, conf, id in zip(boxes, confs, ids):
            obj = Object()
            obj.id = self.yolo.labels[id]
            obj.bbox = [float(coord) for coord in box]
            obj.confidence = float(conf)
            output_msg.objects.append(obj)

        self.yolo.draw_detections(cv_image, boxes, confs, ids)
        image_message = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(image_message)
        self.publisher_.publish(output_msg)

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
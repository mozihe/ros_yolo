import rclpy
from rclpy.node import Node
from interface.srv import MatchObject
from interface.msg import Object, CheckResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import clip
import torch
from PIL import Image as PILImage

class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)

        self.subscription = self.create_subscription(
            CheckResult,
            '/inference/output',
            self.listener_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher_ = self.create_publisher(Object, '/decision/matched_object', 10)

        self.srv = self.create_service(MatchObject, 'match_object', self.handle_service)

        self.bridge = CvBridge()

        self.objects = None
        self.image = None

    def listener_callback(self, msg):
        self.objects = msg.objects

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def handle_service(self, request, response):
        query_text = request.query
        self.get_logger().info(f'Received query: {query_text}')

        if self.objects is None or self.image is None:
            self.get_logger().warning("No objects or image available for matching.")
            response.success = False
            return response

        matched_obj = self.match_objects_to_text(query_text)

        if matched_obj:
            self.publisher_.publish(matched_obj)
            self.get_logger().info(f'Published matched object with id: {matched_obj.id}')
            response.success = True
        else:
            self.get_logger().info('No object matched.')
            response.success = False

        return response

    def match_objects_to_text(self, text):

        if self.objects is None or self.image is None:
            return None

        image_patches = []
        for obj in self.objects:
            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            patch = self.image[y1:y2, x1:x2]
            image_patches.append(PILImage.fromarray(patch))

        preprocessed_images = torch.stack([self.preprocess(patch).to(self.device) for patch in image_patches])

        with torch.no_grad():
            image_features = self.model.encode_image(preprocessed_images)
            text_features = self.model.encode_text(clip.tokenize([text]).to(self.device))

        similarity = (image_features @ text_features.T).squeeze(1)
        best_match_idx = similarity.argmax().item()

        return self.objects[best_match_idx] if self.objects else None

def main(args=None):
    rclpy.init(args=args)
    decision_node = DecisionNode()
    rclpy.spin(decision_node)

    decision_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

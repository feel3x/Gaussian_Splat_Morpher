import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2

class GoProStreamer(Node):
    def __init__(self):
        super().__init__('gopro_streamer')
        
        # Parameters
        self.declare_parameter('stream_url', 'http://10.5.5.9:8080/live/amba.m3u8')
        self.declare_parameter('topic_name', 'camera/image_raw')
        self.declare_parameter('compressed_topic_name', 'camera/image_raw/compressed')
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('show_view', True)

        stream_url = self.get_parameter('stream_url').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        compressed_topic_name = self.get_parameter('compressed_topic_name').get_parameter_value().string_value
        frame_rate = self.get_parameter('frame_rate').get_parameter_value().double_value
        self.show_view = self.get_parameter('show_view').get_parameter_value().bool_value

        # ROS2 publishers
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.compressed_publisher_ = self.create_publisher(CompressedImage, compressed_topic_name, 10)
        self.bridge = CvBridge()

        # OpenCV video capture
        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open GoPro stream at {stream_url}")

        # Timer callback at given frame rate
        timer_period = 1.0 / frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame from GoPro stream")
            return

        # Convert to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher_.publish(msg)

        # Convert to ROS CompressedImage message
        compressed_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpeg')
        self.compressed_publisher_.publish(compressed_msg)

        # Visualization window (only current frame, not saving)
        if self.show_view:
            cv2.imshow("GoPro Stream", frame)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GoProStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

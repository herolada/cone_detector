import math
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

# Poses are in the camera optical frame (x right, y down, z forward). Rotating +90 deg about x
# points the pose's z axis up, along the cone, which is also the axis of the cylinder marker.
UPRIGHT = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # x, y, z, w


class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')
        model_path = self.declare_parameter('model_path', 'cone_yolo.onnx').value
        self.score_thr = self.declare_parameter('score_threshold', 0.4).value
        self.nms_thr = self.declare_parameter('nms_threshold', 0.5).value
        self.class_names = self.declare_parameter('class_names', ['cone']).value
        self.cone_height = self.declare_parameter('cone_height', 0.5).value
        self.rectified = self.declare_parameter('rectified', False).value
        gpu_device = self.declare_parameter('gpu_device', 0).value  # ONNX Runtime CUDA only

        if not os.path.isabs(model_path):  # bare file name -> bundled models/ dir
            model_path = os.path.join(
                get_package_share_directory('cone_detector'), 'models', model_path)
        if model_path.endswith('.engine'):
            from cone_detector.trt_runner import TrtRunner
            self.infer = TrtRunner(model_path)
            backend = 'TensorRT (GPU)'
        else:
            from cone_detector.ort_runner import OrtRunner
            self.infer = OrtRunner(model_path, gpu_device)
            backend = f'ONNX Runtime ({self.infer.provider})'
            if self.infer.cuda_error:
                self.get_logger().warning(
                    f'ONNX Runtime CUDA failed, using CPU: {self.infer.cuda_error}')
        self.in_h, self.in_w = self.infer.input_shape[2:]
        self.bridge = CvBridge()
        self.camera = None  # (K, D, fisheye) from the latest CameraInfo

        self.pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'cone_poses', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'markers', 10)
        self.debug_pub = self.create_publisher(Image, 'debug_image', 1)
        self.create_subscription(CameraInfo, 'camera_info', self.on_camera_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, 'image', self.on_image, qos_profile_sensor_data)
        self.get_logger().info(
            f'Loaded {model_path} ({self.in_w}x{self.in_h}) using {backend}')

    def on_camera_info(self, msg):
        if self.rectified:  # rectified image: projection matrix P, no distortion
            K = np.array(msg.p, float).reshape(3, 4)[:, :3]
            D = np.zeros(4)
        else:
            K = np.array(msg.k, float).reshape(3, 3)
            D = np.array(msg.d, float) if len(msg.d) else np.zeros(4)
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            self.get_logger().warning('CameraInfo has no intrinsics, cannot locate cones',
                                   throttle_duration_sec=10.0)
            return
        self.camera = (K, D, msg.distortion_model == 'equidistant')

    def normalize(self, pts):
        """Pixel points (N, 2) -> undistorted normalized image coordinates (N, 2)."""
        K, D, fisheye = self.camera
        pts = pts.reshape(-1, 1, 2).astype(np.float64)
        if fisheye:
            return cv2.fisheye.undistortPoints(pts, K, D[:4]).reshape(-1, 2)
        return cv2.undistortPoints(pts, K, D).reshape(-1, 2)

    def locate(self, xyxy):
        """Cone mid-height points (N, 3) in the optical frame and base widths (N,).

        Assumes upright cones of height cone_height whose vertical axis maps to the image
        y axis (camera roughly level), so depth = height / normalized box height.
        """
        cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
        top, bottom, left, right = (self.normalize(np.stack(p, 1)) for p in (
            (cx, xyxy[:, 1]), (cx, xyxy[:, 3]), (xyxy[:, 0], cy), (xyxy[:, 2], cy)))
        z = self.cone_height / np.maximum(bottom[:, 1] - top[:, 1], 1e-6)
        mid = (top + bottom) / 2
        return np.column_stack((mid * z[:, None], z)), (right[:, 0] - left[:, 0]) * z

    def decode(self, out):
        """Return (xyxy, scores, class ids) in letterboxed input pixels."""
        if out.shape[0] > out.shape[1]:
            # End-to-end export (NMS inside the model): [x1, y1, x2, y2, score, class] x top-k.
            keep = out[:, 4] > self.score_thr
            return out[keep, :4], out[keep, 4], out[keep, 5].astype(int)

        # Raw head: [cx, cy, w, h, class scores...] x anchors, needs NMS.
        out = out.T
        cls = out[:, 4:].argmax(1)
        scores = out[np.arange(len(out)), 4 + cls]
        keep = scores > self.score_thr
        cxcywh, scores, cls = out[keep, :4], scores[keep], cls[keep]
        xywh = np.hstack((cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, 2:]))
        # Per-class NMS: shift each class's boxes apart (NMSBoxesBatched needs OpenCV >= 4.7).
        shifted = xywh + np.hstack((cls[:, None] * 4096.0, np.zeros((len(cls), 3))))
        idx = np.array(cv2.dnn.NMSBoxes(
            shifted.tolist(), scores.tolist(), self.score_thr, self.nms_thr),
            dtype=int).flatten()
        xyxy = np.hstack((xywh[idx, :2], xywh[idx, :2] + xywh[idx, 2:]))
        return xyxy, scores[idx], cls[idx]

    def make_markers(self, header, i, p, width, rng, bearing):
        cone = Marker(header=header, ns='cones', id=i, type=Marker.CYLINDER)
        cone.pose.position.x, cone.pose.position.y, cone.pose.position.z = map(float, p)
        cone.pose.orientation.x, _, _, cone.pose.orientation.w = UPRIGHT
        cone.scale.x = cone.scale.y = max(float(width), 0.05)
        cone.scale.z = float(self.cone_height)
        cone.color.r, cone.color.g, cone.color.b, cone.color.a = 1.0, 0.45, 0.0, 0.8
        text = Marker(header=header, ns='cone_labels', id=i, type=Marker.TEXT_VIEW_FACING)
        text.pose.position.x, text.pose.position.z = float(p[0]), float(p[2])
        text.pose.position.y = float(p[1] - 0.75 * self.cone_height)  # above the tip
        text.pose.orientation.w = 1.0
        text.scale.z = 0.2
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = f'{rng:.1f} m {math.degrees(bearing):+.0f}°'
        return [cone, text]

    def on_image(self, msg):
        t0 = time.perf_counter()
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Centered gray letterbox, as in Ultralytics training.
        h, w = img.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nh, nw = round(h * r), round(w * r)
        top, left = (self.in_h - nh) // 2, (self.in_w - nw) // 2
        padded = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        padded[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))

        t1 = time.perf_counter()
        out = self.infer(cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True))[0]
        t2 = time.perf_counter()

        xyxy, scores, cls = self.decode(out)
        xyxy = ((xyxy - [left, top, left, top]) / r).clip(0, [w, h, w, h])  # original pixels

        # Boxes cut by the top or bottom image border have a wrong height, so no position.
        located = (xyxy[:, 1] > 1) & (xyxy[:, 3] < h - 2) & (self.camera is not None)
        if self.camera is None and len(xyxy):
            self.get_logger().warning('No CameraInfo received yet, not locating cones',
                                   throttle_duration_sec=10.0)
        points, widths = self.locate(xyxy) if self.camera is not None and len(xyxy) else (
            np.zeros((len(xyxy), 3)), np.zeros(len(xyxy)))
        ranges = np.hypot(points[:, 0], points[:, 2])  # horizontal range
        bearings = np.arctan2(-points[:, 0], points[:, 2])  # positive = left, as ROS yaw

        det = Detection2DArray(header=msg.header)
        markers = MarkerArray(markers=[Marker(header=msg.header, action=Marker.DELETEALL)])
        for i, ((x1, y1, x2, y2), score, c) in enumerate(zip(xyxy, scores, cls)):
            d = Detection2D(header=msg.header)
            d.bbox.center.position.x = float(x1 + x2) / 2
            d.bbox.center.position.y = float(y1 + y2) / 2
            d.bbox.size_x, d.bbox.size_y = float(x2 - x1), float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = (
                self.class_names[c] if c < len(self.class_names) else str(c))
            hyp.hypothesis.score = float(score)
            if located[i]:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, points[i])
                (pose.orientation.x, pose.orientation.y, pose.orientation.z,
                 pose.orientation.w) = UPRIGHT
                hyp.pose.pose = pose
                self.pose_pub.publish(PoseStamped(header=msg.header, pose=pose))
                markers.markers += self.make_markers(
                    msg.header, i, points[i], widths[i], ranges[i], bearings[i])
            d.results.append(hyp)
            det.detections.append(d)
        self.pub.publish(det)
        self.marker_pub.publish(markers)

        if self.debug_pub.get_subscription_count() > 0:
            for (x1, y1, x2, y2), score, ok, rng in zip(
                    xyxy.astype(int), scores, located, ranges):
                label = f'{score:.2f} {rng:.1f}m' if ok else f'{score:.2f}'
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            debug = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            debug.header = msg.header
            self.debug_pub.publish(debug)

        t3 = time.perf_counter()
        self.get_logger().info(
            f'{len(det.detections)} cones ({int(located.sum())} located) | total '
            f'{(t3 - t0) * 1e3:.1f} ms (pre {(t1 - t0) * 1e3:.1f}, '
            f'infer {(t2 - t1) * 1e3:.1f}, post {(t3 - t2) * 1e3:.1f})',
            throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = ConeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

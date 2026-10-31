"""Cone search on top of crl_commander: explore or follow a route, and visit each new cone once.

    BASE --confirmed unchecked cone--> GOTO --within approach_distance--> STOP
    STOP --stop_duration--> mark cone checked --> BASE
    GOTO --goto_timeout--> mark cone checked (unreachable) --> BASE

BASE is EXPLORE (mode 'explore', commander explores) or ROUTE (mode 'route', the route is
requested from commander_operator, again after each cone, the commander then continues from the
nearest waypoint ahead).

Cones are tracked in the map frame. A cone counts once it was detected min_detections times
within association_radius; detections within checked_radius of a checked cone are ignored.
"""
import math

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401, registers PoseStamped with tf2_ros.Buffer.transform
from commander_operator_interfaces.msg import OperatorRequest
from commander_operator_interfaces.srv import SendOperatorRequest
from crl_commander_interfaces.msg import OperatorGoal
from crl_commander_interfaces.srv import SwitchMode
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

EXPLORE, GOTO, STOP = SwitchMode.Request.MODE_EXPLORE, SwitchMode.Request.MODE_GOTO, \
    SwitchMode.Request.MODE_STOP
ROUTE = 'route'


class Cone:
    def __init__(self, xy, weight, t):
        self.xy, self.weight, self.count, self.last_seen = np.asarray(xy, float), weight, 1, t

    def update(self, xy, weight, t):
        self.xy = (self.xy * self.weight + np.asarray(xy) * weight) / (self.weight + weight)
        self.weight += weight
        self.count += 1
        self.last_seen = t


class ConeMemory:
    """ROS-free bookkeeping of cone candidates and checked areas (map-frame x, y)."""

    def __init__(self, association_radius, checked_radius, min_detections, forget_after):
        self.association_radius = association_radius
        self.checked_radius = checked_radius
        self.min_detections = min_detections
        self.forget_after = forget_after
        self.candidates = []
        self.checked = []

    def is_checked(self, xy):
        return any(np.hypot(*(np.asarray(xy) - c)) <= self.checked_radius for c in self.checked)

    def add(self, xy, t, detection_range):
        """Add a detection; close-range ones get more weight, as their distance is better."""
        if self.is_checked(xy):
            return None
        weight = 1.0 / max(detection_range, 1.0) ** 2
        dists = [np.hypot(*(np.asarray(xy) - c.xy)) for c in self.candidates]
        if dists and min(dists) <= self.association_radius:
            cone = self.candidates[int(np.argmin(dists))]
            cone.update(xy, weight, t)
        else:
            cone = Cone(xy, weight, t)
            self.candidates.append(cone)
        return cone

    def prune(self, t):
        self.candidates = [c for c in self.candidates if t - c.last_seen <= self.forget_after]

    def next_target(self, robot_xy):
        """Nearest confirmed, unchecked candidate, or None."""
        confirmed = [c for c in self.candidates
                     if c.count >= self.min_detections and not self.is_checked(c.xy)]
        return min(confirmed, key=lambda c: np.hypot(*(c.xy - robot_xy)), default=None)

    def mark_checked(self, xy):
        self.checked.append(np.asarray(xy, float))
        self.candidates = [c for c in self.candidates if not self.is_checked(c.xy)]


class ConeFinder(Node):
    def __init__(self):
        super().__init__('cone_finder')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.robot_frame = p('robot_frame', 'base_link').value
        self.approach_distance = p('approach_distance', 3.0).value
        self.stop_duration = p('stop_duration', 3.0).value
        self.goto_timeout = p('goto_timeout', 120.0).value
        self.max_detection_range = p('max_detection_range', 20.0).value
        self.goal_update_distance = p('goal_update_distance', 1.0).value
        self.base = p('mode', EXPLORE).value
        if self.base not in (EXPLORE, ROUTE):
            raise ValueError(f"mode must be '{EXPLORE}' or '{ROUTE}', not '{self.base}'")
        self.route_request = OperatorRequest(
            route=True, wgs=p('route_wgs', True).value, goal=p('route_name', 'hrad').value)
        self.memory = ConeMemory(
            association_radius=p('association_radius', 2.0).value,
            checked_radius=p('checked_radius', 3.0).value,
            min_detections=p('min_detections', 3).value,
            forget_after=p('forget_after', 10.0).value)

        self.state = self.base
        self.state_since = self.now()
        self.target = None
        self.sent_goal_xy = None
        # Commander mode we want vs. the one it last acknowledged; synced in tick() so a
        # failed or not-yet-available service is simply retried.
        self.wanted_mode, self.acked_mode, self.request_pending = self.base, None, False

        self.pending_cones = []  # detections waiting for their TF, see transform_cones()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.switch_mode = self.create_client(
            SwitchMode, p('switch_mode_service', '/crl_commander/switch_mode').value)
        self.operator_request = self.create_client(
            SendOperatorRequest,
            p('operator_request_service', '/commander_operator/operator_request').value)
        self.goal_pub = self.create_publisher(
            OperatorGoal, 'operator_goal', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.state_pub = self.create_publisher(String, '~/state', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', 10)
        self.create_subscription(PoseStamped, 'cone_poses', self.on_cone, 10)
        self.create_timer(1.0 / p('rate', 5.0).value, self.tick)
        route = f", route '{self.route_request.goal}'" if self.base == ROUTE else ''
        self.get_logger().info(
            f'Cone finder running in {self.base} mode{route}, cones tracked in {self.map_frame}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.robot_frame, Time())
        except TransformException as e:
            self.get_logger().warning(f'No robot pose: {e}', throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        return np.array([t.x, t.y]), t.z

    def on_cone(self, msg):
        p = msg.pose.position  # camera optical frame: z forward, x right
        if math.hypot(p.x, p.z) <= self.max_detection_range:
            self.pending_cones.append(msg)

    def transform_cones(self):
        """Move pending detections into memory once TF for their stamp is available.

        Never blocks: TF may lag the image a little, so retry on later ticks for up to 1 s.
        """
        t = self.now()
        waiting = []
        for msg in self.pending_cones:
            try:
                cone = self.tf_buffer.transform(msg, self.map_frame)
            except TransformException as e:
                if t - Time.from_msg(msg.header.stamp).nanoseconds * 1e-9 < 1.0:
                    waiting.append(msg)
                else:
                    self.get_logger().warning(f'Cannot transform cone to {self.map_frame}: {e}',
                                           throttle_duration_sec=5.0)
                continue
            p = msg.pose.position
            self.memory.add((cone.pose.position.x, cone.pose.position.y), t, math.hypot(p.x, p.z))
        self.pending_cones = waiting

    def send_goal(self, robot_xy, robot_z):
        """Publish the target as operator goal, facing the cone from the robot."""
        goal = OperatorGoal()
        goal.goal.header.frame_id = self.map_frame
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.pose.position.x, goal.goal.pose.position.y = map(float, self.target.xy)
        goal.goal.pose.position.z = float(robot_z)
        dx, dy = self.target.xy - robot_xy
        yaw = math.atan2(dy, dx)
        goal.goal.pose.orientation.z, goal.goal.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_pub.publish(goal)
        self.sent_goal_xy = self.target.xy.copy()

    def set_state(self, state):
        self.get_logger().info(f'{self.state} -> {state}')
        self.state, self.state_since, self.wanted_mode = state, self.now(), state

    def finish_target(self, reason):
        self.get_logger().info(
            f'Cone at ({self.target.xy[0]:.1f}, {self.target.xy[1]:.1f}) checked ({reason})')
        self.memory.mark_checked(self.target.xy)
        self.target = None
        self.set_state(self.base)

    def sync_mode(self):
        if self.wanted_mode == self.acked_mode or self.request_pending:
            return
        mode = self.wanted_mode
        # The route goes through commander_operator, which also switches the commander mode.
        if mode == ROUTE:
            client, request = self.operator_request, SendOperatorRequest.Request(
                request=self.route_request)
        else:
            client, request = self.switch_mode, SwitchMode.Request(mode=mode)
        if not client.service_is_ready():
            self.get_logger().warning(f'Waiting for the {client.srv_name} service',
                                   throttle_duration_sec=5.0)
            return
        self.request_pending = True

        def done(future):
            self.request_pending = False
            res = future.result()
            if res is not None and res.success:
                self.acked_mode = mode
            else:
                reason = getattr(res, 'message', '') if res else future.exception()
                self.get_logger().error(f'{client.srv_name} {mode} failed: {reason}',
                                        throttle_duration_sec=5.0)

        client.call_async(request).add_done_callback(done)

    def tick(self):
        t = self.now()
        self.transform_cones()
        self.memory.prune(t)
        pose = self.robot_pose()
        if pose is not None:
            robot_xy, robot_z = pose
            if self.state == self.base:
                self.target = self.memory.next_target(robot_xy)
                if self.target is not None:
                    self.get_logger().info(
                        f'New cone at ({self.target.xy[0]:.1f}, {self.target.xy[1]:.1f}), '
                        f'{np.hypot(*(self.target.xy - robot_xy)):.1f} m away')
                    self.send_goal(robot_xy, robot_z)
                    self.set_state(GOTO)
            elif self.state == GOTO:
                if np.hypot(*(self.target.xy - robot_xy)) <= self.approach_distance:
                    self.set_state(STOP)
                elif t - self.state_since > self.goto_timeout:
                    self.get_logger().warning(f'Cone not reached in {self.goto_timeout:.0f} s')
                    self.finish_target('timed out')
                elif np.hypot(*(self.target.xy - self.sent_goal_xy)) > self.goal_update_distance:
                    self.send_goal(robot_xy, robot_z)  # estimate refined as we got closer
            elif self.state == STOP and t - self.state_since >= self.stop_duration:
                self.finish_target('visited')
        self.sync_mode()
        self.state_pub.publish(String(data=self.state))
        self.publish_markers()

    def publish_markers(self):
        stamp = self.get_clock().now().to_msg()
        markers = [Marker(action=Marker.DELETEALL)]

        def marker(ns, i, xy, kind, size, height, rgba):
            m = Marker(ns=ns, id=i, type=kind)
            m.header.frame_id, m.header.stamp = self.map_frame, stamp
            m.pose.position.x, m.pose.position.y = map(float, xy)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = float(size)
            m.scale.z = float(height)
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            return m

        for i, xy in enumerate(self.memory.checked):
            markers.append(marker('checked', i, xy, Marker.CYLINDER,
                                  2 * self.memory.checked_radius, 0.05, (0.0, 0.8, 0.2, 0.3)))
        for i, c in enumerate(self.memory.candidates):
            confirmed = c.count >= self.memory.min_detections
            markers.append(marker('candidates', i, c.xy, Marker.SPHERE, 0.4, 0.4,
                                  (1.0, 0.5, 0.0, 0.9) if confirmed else (1.0, 1.0, 0.0, 0.5)))
        if self.target is not None:
            markers.append(marker('target', 0, self.target.xy, Marker.CYLINDER,
                                  2 * self.approach_distance, 0.02, (1.0, 0.0, 0.0, 0.25)))
        self.marker_pub.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    node = ConeFinder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

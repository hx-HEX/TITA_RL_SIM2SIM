#!/usr/bin/env python3
import sys, termios, tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def get_key():
    """读取单个按键"""
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


settings = termios.tcgetattr(sys.stdin)


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 参数：可调速度
        self.linear_speed = 0.5   # 前进速度
        self.angular_speed = 0.5  # 横向速度
        self.height_speed = 0.0  # 高度调节速度

        self.get_logger().info(
            "\n控制说明:\n"
            "W/S: 前进 / 后退\n"
            "A/D: 左移 / 右移\n"
            "R/F: 升高 / 降低\n"
            "Q  : 退出\n"
        )

    def run(self):
        twist = Twist()
        while rclpy.ok():
            key = get_key()

            if key == 'w':
                twist.linear.x = self.linear_speed
            elif key == 's':
                twist.linear.x = -self.linear_speed
            elif key == 'a':
                twist.angular.z= self.angular_speed
            elif key == 'd':
                twist.angular.z = -self.angular_speed
            # elif key == 'r':
            #     twist.angular.z += self.height_speed
            # elif key == 'f':
            #     twist.angular.z -= self.height_speed
            elif key == 'q':
                self.get_logger().info("退出键盘控制")
                break
            else:
                # 松开键时复位速度（高度保持累积值）
                twist.linear.x = 0.0
                twist.linear.y = 0.0
                twist.angular.z = 0.0
            self.pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

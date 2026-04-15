#!/usr/bin/env python3
"""Mecanum Teleop — "drive the robot by hand."

A small tkinter window with three sliders (``vx``, ``vy``, ``omega``) that
publish directly to ``WSKR/cmd_vel``. The autopilot is bypassed entirely —
your slider values go straight to the serial bridge and out to the base.

Use it for:
    - Verifying the Arduino firmware responds to ``V`` commands correctly.
    - Testing that the four wheels go the right direction for each input
      axis before you trust the autopilot.
    - Driving a tag into and out of the FOV to exercise the
      visual/dead_reckoning handoff in ``dead_reckoning_node``.

The sliders snap back to zero on mouse release by default (safety). Live
``/odom`` pose and ``arduino/status`` messages are shown in the side panel.
"""
from __future__ import annotations

import math
import threading
from typing import Optional

import rclpy
import tkinter as tk
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Empty, String


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MecanumTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__('mecanum_teleop')

        self._lock = threading.Lock()
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0

        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw_deg = 0.0
        self._odom_vx = 0.0
        self._odom_vy = 0.0
        self._odom_omega_degs = 0.0

        self._status_lines: list[str] = []

        self._cmd_pub = self.create_publisher(Twist, 'WSKR/cmd_vel', 10)
        self._stop_pub = self.create_publisher(Empty, 'WSKR/stop', 1)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(String, 'arduino/status', self._on_status, 10)

        # Publish at 20 Hz — well within the bridge's 0.5 s cmd_timeout
        self.create_timer(0.05, self._publish_cmd)

        threading.Thread(target=self._run_gui, daemon=True).start()
        self.get_logger().info('Mecanum teleop ready. Publishing to WSKR/cmd_vel.')

    # ------------------------------------------------------------------ ROS

    def _publish_cmd(self) -> None:
        with self._lock:
            vx, vy, omega = self._vx, self._vy, self._omega
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = omega
        self._cmd_pub.publish(msg)

    def _on_odom(self, msg: Odometry) -> None:
        yaw = _quat_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        with self._lock:
            self._odom_x = msg.pose.pose.position.x
            self._odom_y = msg.pose.pose.position.y
            self._odom_yaw_deg = math.degrees(yaw)
            self._odom_vx = msg.twist.twist.linear.x
            self._odom_vy = msg.twist.twist.linear.y
            self._odom_omega_degs = math.degrees(msg.twist.twist.angular.z)

    def _on_status(self, msg: String) -> None:
        with self._lock:
            self._status_lines.append(msg.data)
            if len(self._status_lines) > 8:
                self._status_lines = self._status_lines[-8:]

    # ------------------------------------------------------------------ GUI

    def _run_gui(self) -> None:
        BG = '#1e1e1e'
        PANEL_BG = '#2a2a2a'
        FG = 'white'
        ACCENT = '#80d0ff'
        VALUE_FG = '#ffcc00'
        FONT = ('Arial', 11)
        MONO = ('Consolas', 11)

        root = tk.Tk()
        root.title('Mecanum Teleop')
        root.configure(bg=BG)
        root.resizable(False, False)

        def lf(parent: tk.Widget, title: str) -> tk.LabelFrame:
            return tk.LabelFrame(
                parent, text=title, bg=BG, fg=ACCENT,
                font=('Arial', 12, 'bold'), padx=10, pady=6,
            )

        # ---- Snap-back toggle ----
        snap_var = tk.BooleanVar(value=True)
        snap_frame = tk.Frame(root, bg=BG)
        snap_frame.pack(fill=tk.X, padx=14, pady=(10, 0))
        tk.Checkbutton(
            snap_frame, text='Snap sliders to zero on release (recommended)',
            variable=snap_var, bg=BG, fg='#c0c0c0', selectcolor=PANEL_BG,
            activebackground=BG, activeforeground=FG, font=('Arial', 10),
        ).pack(side=tk.LEFT)

        # ---- Sliders ----
        ctrl = lf(root, 'Drive Command')
        ctrl.pack(fill=tk.X, padx=12, pady=(6, 4))

        SLIDER_DEFS = [
            ('Forward / Back  (m/s)', -0.5,  0.5),
            ('Strafe  L / R   (m/s)', -0.5,  0.5),
            ('Turn           (rad/s)', -1.5,  1.5),
        ]
        slider_vars: list[tk.DoubleVar] = []
        disp_vars: list[tk.StringVar] = []

        for row_idx, (label, lo, hi) in enumerate(SLIDER_DEFS):
            tk.Label(
                ctrl, text=label, bg=BG, fg=FG, font=FONT, width=24, anchor='e',
            ).grid(row=row_idx, column=0, padx=(0, 8), pady=5)

            var = tk.DoubleVar(value=0.0)
            slider_vars.append(var)

            sl = tk.Scale(
                ctrl, variable=var, from_=lo, to=hi, resolution=0.01,
                orient=tk.HORIZONTAL, length=300, bg=PANEL_BG, fg=FG,
                troughcolor='#555', activebackground='#5599ff',
                highlightthickness=0, showvalue=False,
            )
            sl.grid(row=row_idx, column=1, padx=4, pady=5)

            def _on_release(event, sv=var):
                if snap_var.get():
                    sv.set(0.0)

            sl.bind('<ButtonRelease-1>', _on_release)

            dv = tk.StringVar(value=' 0.00')
            disp_vars.append(dv)
            tk.Label(
                ctrl, textvariable=dv, bg=BG, fg=VALUE_FG, font=MONO, width=7,
            ).grid(row=row_idx, column=2, padx=(6, 0))

        # ---- Stop button ----
        stop_row = tk.Frame(root, bg=BG)
        stop_row.pack(fill=tk.X, padx=12, pady=4)

        def do_stop() -> None:
            for sv in slider_vars:
                sv.set(0.0)
            with self._lock:
                self._vx = 0.0
                self._vy = 0.0
                self._omega = 0.0
            self._stop_pub.publish(Empty())

        tk.Button(
            stop_row, text='STOP', command=do_stop,
            bg='#c62828', fg='white', font=('Arial', 16, 'bold'),
            padx=40, pady=8, activebackground='#8b0000',
        ).pack()

        # ---- Odometry panel ----
        odom_frame = lf(root, 'Odometry  (/odom)')
        odom_frame.pack(fill=tk.X, padx=12, pady=4)

        odom_keys = ('pos_x', 'pos_y', 'yaw', 'vel_vx', 'vel_vy', 'vel_omega')
        odom_vars: dict[str, tk.StringVar] = {k: tk.StringVar(value='—') for k in odom_keys}

        ODOM_DEFS = [
            ('Pos X (m)',    'pos_x'),
            ('Pos Y (m)',    'pos_y'),
            ('Yaw (°)',      'yaw'),
            ('Vel vx (m/s)', 'vel_vx'),
            ('Vel vy (m/s)', 'vel_vy'),
            ('Omega (°/s)',  'vel_omega'),
        ]
        # Laid out as two rows of three label+value pairs
        for i, (lbl, key) in enumerate(ODOM_DEFS):
            col = (i % 3) * 2
            row = i // 3
            tk.Label(
                odom_frame, text=lbl, bg=BG, fg=FG, font=FONT, anchor='e', width=14,
            ).grid(row=row, column=col, padx=(0, 4), pady=3, sticky='e')
            tk.Label(
                odom_frame, textvariable=odom_vars[key], bg=BG, fg=VALUE_FG,
                font=MONO, width=10, anchor='w',
            ).grid(row=row, column=col + 1, padx=(0, 14), pady=3, sticky='w')

        # ---- Arduino status log ----
        status_lf = lf(root, 'Arduino Status  (arduino/status)')
        status_lf.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))

        status_text = tk.Text(
            status_lf, height=6, bg=PANEL_BG, fg='#c0c0c0',
            font=MONO, state=tk.DISABLED, relief=tk.FLAT, bd=0,
        )
        status_text.pack(fill=tk.BOTH, expand=True)

        # ---- Refresh loop (~20 Hz) ----
        def refresh() -> None:
            # Read slider values and push to shared state
            vx = slider_vars[0].get()
            vy = slider_vars[1].get()
            omega = slider_vars[2].get()
            with self._lock:
                self._vx = vx
                self._vy = vy
                self._omega = omega
                x = self._odom_x
                y = self._odom_y
                yaw = self._odom_yaw_deg
                ovx = self._odom_vx
                ovy = self._odom_vy
                oomega = self._odom_omega_degs
                lines = list(self._status_lines)

            disp_vars[0].set(f'{vx:+.2f}')
            disp_vars[1].set(f'{vy:+.2f}')
            disp_vars[2].set(f'{omega:+.2f}')

            odom_vars['pos_x'].set(f'{x:+.3f}')
            odom_vars['pos_y'].set(f'{y:+.3f}')
            odom_vars['yaw'].set(f'{yaw:+.1f}')
            odom_vars['vel_vx'].set(f'{ovx:+.3f}')
            odom_vars['vel_vy'].set(f'{ovy:+.3f}')
            odom_vars['vel_omega'].set(f'{oomega:+.1f}')

            status_text.config(state=tk.NORMAL)
            status_text.delete('1.0', tk.END)
            status_text.insert(tk.END, '\n'.join(lines))
            status_text.config(state=tk.DISABLED)

            root.after(50, refresh)

        root.after(50, refresh)

        def on_close() -> None:
            with self._lock:
                self._vx = 0.0
                self._vy = 0.0
                self._omega = 0.0
            self._stop_pub.publish(Empty())
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', on_close)
        root.mainloop()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MecanumTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

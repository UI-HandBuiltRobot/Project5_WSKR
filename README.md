# Project 5 — Vision-Based Robot Navigation

> **Before doing anything else, install the required Python packages:**
> ```bash
> pip install -r project5_ws/requirements.txt
> ```
> This pins NumPy and OpenCV to versions that are compatible with ROS 2's
> `cv_bridge`. Using different versions will cause runtime import errors.

This workspace contains the ROS 2 code and the GUI tools for the
vision-based mecanum robot. Teams supply their own Arduino sketch
(modified per the spec in [The Arduino side](#the-arduino-side)) to
drive the mecanum base. The goal of the system is simple: **point the
robot's camera at an ArUco tag and have the robot autonomously drive up
to it while avoiding obstacles.**

At the highest level, three things have to work together:

1. **Perception** — turn raw camera frames into two numbers per tick:
   "how far can I drive in each direction?" (whiskers) and "where is my
   target?" (heading to tag).
2. **The autopilot** — a small neural network that turns those numbers
   into a drive command (Twist: forward / strafe / rotate).
3. **The base** — an Arduino on a mecanum chassis that executes that
   drive command and reports odometry back so we can keep tracking the
   target even when it briefly leaves the camera's field of view.

This README covers how to launch the stack, the calibration workflow,
the utilities you'll use for demos and tuning, and what the Arduino has
to do.

---

## Repository layout

```
Project5_WSKR/
├── project5_ws/                       <- the ROS 2 workspace
│   ├── standalone_python_scripts/
│   │   └── calibrator/
│   │       └── whisker_calibration_tool.py  <- one-time pixel→mm mapper
│   └── src/
│       ├── wskr/                      <- floor mask, whiskers, approach action, DR fuser
│       ├── arduino/                   <- ROS ↔ Arduino serial bridge
│       ├── gstreamer_camera/          <- publishes camera1/image_raw/compressed
│       ├── robot_interfaces/          <- custom .msg / .srv / .action definitions
│       └── utilities/                 <- GUI tools: dashboard, tuners, teleop
```

Each ROS 2 node has a plain-language docstring at the top of its file —
read those first for context on what a specific node does.

---

## Building the workspace

From `project5_ws/`:

```bash
# one-time dependency install (Jetson / Ubuntu 22.04)
sudo apt install python3-serial ros-humble-tf2-ros ros-humble-cv-bridge

# build everything
colcon build --symlink-install

# source the overlay (every new shell)
source install/setup.bash
```

`--symlink-install` means you can edit Python files in `src/` and just
re-run nodes without rebuilding. You do need to rebuild after editing
`CMakeLists.txt`, `setup.py`, or a message/service/action definition.

---

## Getting the system running — recommended workflow

Follow these steps in order the first time you set up a new robot.

### Step 1 — Launch the WSKR stack

The main launch file brings up **everything** needed for the autonomous
approach behavior — camera, floor finder, whisker computer, approach
action server, dead-reckoning fuser, autopilot, and the Arduino serial
bridge:

```bash
ros2 launch wskr wskr.launch.py
```

Optional argument: `camera_rate_hz:=10.0` throttles the camera
publication rate. Drop it lower (e.g. `camera_rate_hz:=5.0`) if the
Jetson is thermally throttling or if you want to save USB bandwidth.

**What comes up:**

| Node | What it does |
|---|---|
| `gstreamer_camera` | Publishes JPEG frames from the USB camera on `camera1/image_raw/compressed`. |
| `wskr_floor` | Finds the floor in each frame → `WSKR/floor_mask`. |
| `wskr_range` | Walks 11 whisker rays on the mask → `WSKR/whisker_lengths` and `WSKR/target_whisker_lengths`. Also publishes the composite `wskr_overlay/compressed`. |
| `wskr_approach_action` | Supervises approach goals: runs ArUco/CSRT tracking, publishes a visual heading observation, enables the autopilot for the duration of the goal, and decides when the target is reached. |
| `wskr_dead_reckoning` | Fuses visual heading and odometry yaw into a single `WSKR/heading_to_target` with hysteresis. |
| `wskr_autopilot` | Runs the MLP inference loop, converting whisker + heading observations into `WSKR/cmd_vel` Twist commands. Gated by `WSKR/autopilot/enable`. |
| `mecanum_serial_bridge` | Translates `WSKR/cmd_vel` ↔ Arduino serial, publishes `/odom`. |

### Step 2 — Tune the floor mask (`floor_tuner`)

```bash
# in a new terminal, with the stack running:
ros2 run utilities floor_tuner
```

The floor mask is produced by thresholding color similarity against a
sample patch near the bottom of the image, plus a gradient/edge test.
Lighting and floor material make a big difference.

Slide until the green-tinted overlay on the right looks right →
click **Apply** to push the values into the running `wskr_floor` node →
**Save YAML** to persist them to `config/floor_params.yaml` for next
launch.

Use this when the floor mask is noisy or has holes, you've moved to a
new room, or the whiskers look unreasonably short or long.

### Step 3 — Tune the heading meridians (`heading_tuner`)

```bash
ros2 run utilities heading_tuner
```

Heading-to-target is computed from the bbox center's pixel position by
inverting a fisheye lens model. If the camera is tilted slightly
differently from what `lens_params.yaml` assumes, the dashed meridians
won't line up with real-world vertical lines.

Slide `y_offset` until the meridians visually line up with real vertical
lines in the scene → **Save** to persist to `config/lens_params.yaml`.
The value is also pushed live to the running `wskr_range` and
`wskr_approach_action` nodes, so the stack picks it up without a
restart.

Use this when you've remounted the camera, or the robot consistently
turns past the target before re-acquiring.

### Step 4 — Run the whisker calibration tool

The whisker calibration maps each whisker ray's pixel coordinates to
real-world millimetre distances. **Kill the `wskr.launch.py` terminal
first** — the calibration tool opens the camera directly and will
conflict with the running stack.

```bash
cd "project5_ws/standalone_python_scripts/calibrator"
python3 whisker_calibration_tool.py
```

Flow:
- Place the robot in front of a calibration mat or marked reference.
- Snap a frame (or load a previously-saved `.png`).
- Click **Select whisker -30°** (or whichever angle you're calibrating),
  click the matching point on the image, and enter the real-world
  distance in mm. Repeat for several distances per whisker.
- Click **Done** to fit a spline; repeat for all 11 whisker angles.
- Click **Save Calibration** and navigate to
  `project5_ws/src/wskr/config/` — overwrite `whisker_calibration.json`
  with your new file.

After saving, rebuild the `wskr` package so the install-share copy is
refreshed:

```bash
cd project5_ws
colcon build --packages-select wskr
source install/setup.bash
```

Re-run this calibration whenever you change the camera mounting height
or angle, or when whisker distances are systematically wrong.

### Step 5 — Launch the dashboard and start approaching

Re-launch the stack, then open the dashboard:

```bash
# Terminal 1:
ros2 launch wskr wskr.launch.py

# Terminal 2:
ros2 run utilities wskr_dashboard
```

The dashboard is the **command centre** for the entire navigation stack.
It combines monitoring and approach control in one window:

**Tile 1 — ArUco Approach (top-left)**
Live camera preview with locally-detected ArUco markers drawn in grey
(any visible tag) or cyan (the target tag once a goal is active).
Controls: ArUco ID entry, **Start Approach** and **Cancel** buttons,
tag-visible status, and a live feedback row showing tracking mode /
heading / visual lock / closest whisker distance while a goal is
running.

**Tile 2 — WSKR Overlay (top-right)**
The `wskr_overlay/compressed` feed — floor mask in the background,
yellow whisker rays, dashed red heading meridians, and a text strip with
heading / mode / cmd_vel. Obstacle whisker hit points appear as red
dots; target whisker intercepts appear as magenta dots (see
[Dual-whisker system](#dual-whisker-system) below).

**Tile 3 — Telemetry (bottom)**
Fused heading in degrees (green = visual, orange = dead-reckoning), a
top-down robot schematic with the full whisker fan (length ∝ drive
distance, colour-coded red/amber/blue), magenta diamond markers at the
target intercept distance on each ray, a heading arrow, and the latest
autopilot `cmd_vel`.

**To start an approach:**
1. Type your target tag's ID in the yellow ID entry box.
2. Wait for the status to show the tag is visible.
3. Click **Start Approach**.
4. Use **Cancel** to abort.

ArUco tags use dictionary `DICT_4X4_50`. Tags will be shared in class,
but you can also generate your own at <https://chev.me/arucogen/>
(select 4×4, dictionary 50).

---

## Dual-whisker system

`wskr_range` computes **two independent distances per ray** on every frame:

| Topic | What it measures |
|---|---|
| `WSKR/whisker_lengths` | Distance to the first **non-floor pixel** (obstacle or frame edge). |
| `WSKR/target_whisker_lengths` | Distance to the first pixel that falls **inside the tracked ArUco bbox**, or max-range if no fresh bbox is cached. |

This lets the MLP distinguish between "there is an obstacle ahead" and
"that obstacle IS the thing I'm supposed to drive toward." On the
overlay, floor-whisker hit points are drawn as **red dots** and
target-whisker intercepts as **magenta dots**. In the telemetry tile,
magenta diamond markers appear on the whisker fan arms at the target
intercept distance.

---

## Training a navigation model

The autopilot is a small MLP (multi-layer perceptron) whose inputs are
the 11 floor-whisker lengths, the 11 target-whisker lengths, and the
current heading to target, and whose outputs are `vx`, `vy`, and `omega`
velocity commands.

To train a model for your robot's specific geometry and environment, use
the **RoboSim Navigation Trainer** simulator:

<https://github.com/UI-HandBuiltRobot/RoboSim_Navigation_Trainer>

See the simulator's own documentation for full usage details. Once you
have trained a model:

1. Export it as a `.json` file from the trainer.
2. Place it in `project5_ws/src/wskr/wskr/models/`.
   **The models folder should contain exactly one `.json` file** — the
   node loads whichever file it finds there at startup.
3. Rebuild the package:
   ```bash
   colcon build --packages-select wskr
   source install/setup.bash
   ```

---

## Other tuning tools

### `mecanum_teleop` — drive by hand

```bash
ros2 run utilities mecanum_teleop
```

Sliders for `vx`, `vy`, `omega` that publish directly to `WSKR/cmd_vel`,
bypassing the autopilot entirely. Live `/odom` readout in the side
panel.

Use this when:
- You just flashed new firmware and want to verify the `V` command works.
- You want to exercise the visual / dead_reckoning handoff without
  running the MLP.
- You're debugging motor polarity or wheel direction.

Sliders snap back to zero on mouse release for safety — uncheck the
"Snap back" box only if you know what you're doing.

---

## The Arduino side

**This is the most substantial implementation task on the firmware side.
Read this section carefully.**

The mecanum base is driven by an Arduino sketch that **your team is
expected to bring with you** from earlier coursework. The ROS 2 side
assumes the sketch exposes:

- Four-wheel closed-loop speed control at ~20 Hz.
- Full mecanum inverse kinematics (body twist → per-wheel rad/s).
- A CSV serial command protocol at 115200 baud:
  - `V,vx_cm_s,vy_cm_s,omega_deg_s\n` — continuous velocity stream.
  - `X\n` — emergency stop.
- Continuous odometry telemetry after every control tick:
  `O,dyaw_deg,dx_cm,dy_cm,dt_ms\n`.

### You will need to update your own sketch

No reference sketch ships with this repo — every team's mecanum
controller looks a little different, and that's fine. The task here is
to **modify the mecanum wheel controller you already wrote** so it
accepts this project's new input format (`V` / `X` commands) and emits
the `O` odometry telemetry the ROS stack needs.

**If your existing sketch only supports distance-based commands (for
example, "drive forward 30 cm at 15 cm/s"), the ROS side will not be
able to talk to it out of the box.** You must add the `V` streaming
command and the `O` odometry telemetry described above. Without them,
`WSKR/cmd_vel` has nowhere to go and `wskr_dead_reckoning` can't update
when the tag leaves the FOV.

To make that retrofit easier, the prompt below is included so you can
hand it, together with your own `.ino`, to a coding agent (Claude Code,
Cursor, etc.). It is written so an agent can do the modification in one
pass without removing anything you already have:

> You are modifying an existing Arduino sketch that drives a
> mecanum-wheeled robot base. Your goal is to add a new velocity
> streaming command and an odometry telemetry response, while leaving
> the existing command parsing untouched.
>
> ## Context
>
> A ROS2 node will communicate with this Arduino over USB serial at
> 115200 baud. It will:
> - Stream velocity commands at ~20 Hz
> - Send a stop command on shutdown
> - Read odometry telemetry lines that you emit
>
> Do not remove or alter any existing command parsing logic. Add the
> new functionality alongside it.
>
> ## Serial interface spec
>
> ### Commands you must handle (new)
>
> **Velocity command:**
> `V,<vx_cm/s>,<vy_cm/s>,<omega_deg/s>\n`
>
> - vx_cm/s: forward/backward speed in cm/s (positive = forward)
> - vy_cm/s: lateral strafe speed in cm/s (positive = left)
> - omega_deg/s: rotational rate in deg/s (positive = counter-clockwise)
> - All values are signed floating point
> - These commands arrive continuously at ~20 Hz. Store the latest values and
>   apply them each control loop iteration.
> - If no V command has been received for more than 500 ms, stop all motors.
>
> **Stop command:**
> `X\n`
> - Immediately stop all motors and zero the stored velocity setpoints.
>
> ### Telemetry you must emit (new)
>
> After every control loop iteration, emit one odometry line:
> `O,<dyaw_deg>,<dx_cm>,<dy_cm>,<dt_ms>\n`
>
> - dyaw_deg: yaw angle change this iteration, in degrees (positive = CCW)
> - dx_cm: forward displacement this iteration in cm, in the robot's local frame
> - dy_cm: lateral displacement this iteration in cm (positive = left)
> - dt_ms: elapsed time for this loop iteration in milliseconds
> - All values are floating point. Use at least 4 decimal places.
> - Derive these values from wheel encoder deltas and your kinematic model.
> - dt_ms must reflect actual measured loop time, not a nominal value.
>
> Any other status messages you wish to emit (e.g. "Ready", error strings)
> should be sent as plain text lines. The ROS2 node will forward them as
> status strings.
>
> ## Implementation requirements
>
> 1. Parse incoming serial data line-by-line (newline delimited).
> 2. Dispatch on the first character or token: 'V' for velocity, 'X' for stop.
>    All other tokens should be handled by your existing parsing logic unchanged.
> 3. Use a non-blocking serial read pattern - do not use blocking Serial.readString()
>    or similar calls in the main loop.
> 4. Track loop timing using micros() or millis(). Compute dt_ms as the measured
>    elapsed time since the previous iteration.
> 5. Compute encoder-based odometry deltas each loop and emit the O line immediately
>    after applying motor commands.
> 6. Apply motor commands using whatever motor driver interface already exists in
>    the sketch. Map vx, vy, omega to individual wheel speeds using the mecanum
>    kinematic equations:
>      FL = vx - vy - omega * k
>      FR = vx + vy + omega * k
>      RL = vx + vy - omega * k
>      RR = vx - vy + omega * k
>    where k is a tuning constant accounting for wheel geometry. If the sketch
>    already has a mecanum mixing function, use it.
>
> ## What to preserve
>
> - All existing command types and their parsing logic must continue to work
>   exactly as before.
> - Do not change baud rate, pin assignments, or any hardware configuration.
> - Do not change any existing variable names or function signatures.
>
> ## Output
>
> Provide the complete modified sketch. Add a comment block at the top of any
> new functions you add, briefly explaining their purpose. Do not add comments
> to existing code.

### Verifying the Arduino side

After flashing, open a plain serial monitor at 115200 baud and:

1. You should see `O,...` telemetry lines streaming at ~20 Hz.
2. Type `V,0,0,30` + Enter — the robot should rotate CCW slowly. Send
   `V,0,0,0` or `X` to stop.
3. Type `V,10,0,0` + Enter — the robot should creep forward.
4. Confirm the `dyaw`, `dx`, `dy` deltas in the `O` lines match the
   direction of motion.

Only then should you launch the ROS stack and expect the autopilot to
work.

---

## Troubleshooting checklist

| Symptom | Look at |
|---|---|
| Camera tile blank in dashboard | `ls /dev/video*`; check `gstreamer_camera` logs. |
| Floor mask empty or noisy | `floor_tuner` — `color_distance_threshold`, `val_range`. |
| Whiskers all show max range | Floor mask may be all-white (nothing detected as non-floor). Check `floor_tuner`. |
| Target whiskers all at max range | Is a goal active? Target whiskers only register when the approach server is publishing a tracked bbox. |
| Heading stays pinned at 0 when tag is visible | Is the approach action actually running? Check the dashboard status bar. |
| Robot drives through the target | Check `proximity_success_mm` and whisker calibration (`config/whisker_calibration.json`). |
| Robot does nothing after "Start Approach" | Is `mecanum_serial_bridge` connected to the right `port`? Check `ros2 topic echo /odom` — if nothing, the Arduino isn't talking. |
| Mode flips rapidly between visual and dead_reckoning | Hysteresis too tight — check `dr_handoff_deg` / `visual_reacquire_deg` params on `wskr_dead_reckoning`. |
| "ApproachObject action server not available" in dashboard | `wskr_approach_action` node didn't start — check the launch terminal for errors. |

# Project 5 — Vision-Based Robot Navigation

This workspace contains the ROS 2 code, the Arduino firmware, and the GUI
tools for the vision-based mecanum robot. The goal of the system is
simple: **point the robot's camera at an ArUco tag and have the robot
autonomously drive up to it while avoiding obstacles.**

At the highest level, three things have to work together:

1. **Perception** — turn raw camera frames into two numbers per tick:
   "how far can I drive in each direction?" (whiskers) and "where is my
   target?" (heading to tag).
2. **The autopilot** — a small neural network that turns those numbers
   into a drive command (Twist: forward / strafe / rotate).
3. **The base** — an Arduino on a mecanum chassis that executes that
   drive command and reports odometry back so we can keep tracking the
   target even when it briefly leaves the camera's field of view.

This README covers how to launch the stack, the utilities you'll use for
demos and tuning, and what the Arduino has to do.

---

## Repository layout

```
Project5_Solution_Vision_Navigation/
├── Project5_WS/                       <- the ROS 2 workspace
│   ├── Arduino/
│   │   └── Mecanum_Wheel_Controller_Compact/
│   │       └── Mecanum_Wheel_Controller_Compact.ino
│   └── src/
│       ├── WSKR/                      <- floor mask, whiskers, approach action, DR fuser
│       ├── arduino/                   <- ROS <-> Arduino serial bridge
│       ├── camera_processes/
│       ├── gstreamer_camera/          <- the one that publishes camera1/image_raw/compressed
│       ├── robot_interfaces/          <- custom .msg / .srv / .action definitions
│       ├── system_manager_package/    <- high-level state machine (pick-and-drop workflow)
│       ├── utilities/                 <- GUI tools: approach, dashboard, tuners, teleop
│       └── vision_processing_package/ <- Roboflow-based object detection service chain
└── Standalone Python Scripts/         <- one-off helpers, not part of the ROS build
```

Each ROS 2 node has a plain-language docstring at the top of its file —
read those first for context on what a specific node does.

---

## Building the workspace

From `Project5_WS/`:

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

## Launching the WSKR algorithm

The main launch file brings up **everything** needed for the autonomous
approach behavior — camera, floor finder, whisker computer, approach
action server, dead-reckoning fuser, and the Arduino serial bridge:

```bash
ros2 launch wskr wskr.launch.py
```

Optional argument: `camera_rate_hz:=10.0` throttles the camera
publication rate. Drop it lower (e.g. `camera_rate_hz:=5.0`) if the
Jetson is thermally throttling or if you want to save USB bandwidth.

### What comes up

| Node | What it does |
|---|---|
| `gstreamer_camera` | Publishes JPEG frames from the USB camera on `camera1/image_raw/compressed`. |
| `WSKR_floor` | Finds the floor in each frame → `WSKR/floor_mask`. |
| `WSKR_range` | Walks 11 whisker rays on the mask → `WSKR/whisker_lengths`. Also publishes the composite `wskr_overlay/compressed`. |
| `WSKR_approach_action` | Runs ArUco detection, feeds whiskers + heading into the MLP, publishes `WSKR/cmd_vel`. |
| `WSKR_dead_reckoning` | Fuses visual heading and odometry yaw into a single `WSKR/heading_to_target` with hysteresis. |
| `mecanum_serial_bridge` | Translates `WSKR/cmd_vel` ↔ Arduino serial, publishes `/odom`. |

Nothing will move until a client sends an `ApproachObject` goal. Use the
GUI below to do that.

---

## Starting an approach — `Start_Aruco_Approach`

This is the GUI a user interacts with during a demo. It shows a live
camera preview with every ArUco tag it can see overlaid in grey, and the
tag you typed in — if it's visible — highlighted in bright yellow. Hit
the big green "Start Approach" button and the robot will drive up to
that tag.

```bash
# in a new terminal, after wskr.launch.py is running:
ros2 run utilities Start_Aruco_Approach
```

- Type your target tag's ID in the big yellow ID field.
- Wait until the tag shows as `ID N visible @ x=…` (i.e. the preview
  highlighted it).
- Click **Start Approach**. The cancel button turns red; use it to abort.
- The status bar at the bottom shows the current `tracking_mode`
  (visual / dead_reckoning) and fused heading so you can sanity-check
  what the stack is doing.

Only ArUco dictionary `DICT_4X4_50` is recognised. Print tags from
<https://chev.me/arucogen/> (4x4, dictionary 50).

---

## Watching the stack — `WSKR_Dashboard`

A read-only monitoring window. Launch it beside `Start_Aruco_Approach`
to see what each layer of the stack is doing at once:

```bash
ros2 run utilities WSKR_Dashboard
```

Tiles:

1. **Target Tracking** — the main camera feed with the approach server's
   tracked bbox overlaid (drawn only while an action is running).
2. **WSKR Overlay** — the floor mask + labelled whiskers + dashed
   heading meridians + a text strip with mode/heading/cmd_vel.
3. **Telemetry** — a large `heading_to_target` readout (green in visual
   mode, orange in dead-reckoning), 11-bar chart of whisker drive
   distances (red < 150 mm, amber < 400 mm, blue otherwise), and the
   current autopilot Twist.

The dashboard doesn't command anything — it's purely diagnostic.

---

## Tuning tools (`src/utilities/`)

The demo works out of the box only if the calibration matches your
hardware. These GUIs let you fix that without editing code.

### `Floor_Tuner` — what counts as floor?

```bash
ros2 run utilities Floor_Tuner
```

The floor mask is produced by thresholding color similarity against a
sample patch near the bottom of the image, plus a gradient/edge test.
Lighting and floor material make a big difference.

Use this when:
- The floor mask is noisy or has holes.
- You've moved to a new room / different floor surface.
- The whiskers look unreasonably short or long.

Flow: slide until the green-tinted overlay on the right looks right →
click **Apply to /WSKR node** to push the values into the running
`WSKR_floor` node → **Save YAML** to persist them to
`config/floor_params.yaml` for next launch.

### `Heading_Tuner` — why are my meridians crooked?

```bash
ros2 run utilities Heading_Tuner
```

Heading-to-target is computed from the bbox center's pixel position by
inverting a fisheye lens model. If the camera is tilted slightly
differently from what `lens_params.yaml` assumes, the dashed meridians
won't line up with real-world vertical lines.

The window subscribes to `camera1/image_raw/compressed` directly and
draws the meridians on top of the live feed itself — so the meridians
shift **immediately** as you drag the `y_offset` slider (no round trip
through `WSKR_range`). Click anywhere on the preview to get the
computed heading for that pixel rendered as a yellow overlay; useful
for spot-checking that a known vertical in the scene maps to the
heading you expect.

Use this when:
- You've remounted the camera.
- The robot consistently turns past the target before re-acquiring.
- You want to sanity-check the lens model at specific pixels.

Flow: slide `y_offset` until the meridians visually line up with real
vertical lines in the scene → **Save** to persist to
`config/lens_params.yaml`. The value is also pushed live to the
running `WSKR_range` and `WSKR_approach_action` nodes over
`SetParameters`, so the stack picks it up without a restart.

### `Mecanum_Teleop` — drive by hand

```bash
ros2 run utilities Mecanum_Teleop
```

Sliders for `vx`, `vy`, `omega` that publish directly to `WSKR/cmd_vel`,
bypassing the autopilot entirely. Live `/odom` readout in the side
panel.

Use this when:
- You just flashed new firmware and want to verify the `V` command
  works.
- You want to exercise the visual / dead_reckoning handoff without
  running the MLP.
- You're debugging motor polarity or wheel direction.

Sliders snap back to zero on mouse release for safety — uncheck the
"Snap back" box only if you know what you're doing.

### `whisker_calibration_tool` — how many mm is this pixel row?

```bash
cd "Project5_WS/Standalone Python Scripts/Calibrator"
python3 whisker_calibration_tool.py
```

This is the odd one out — it lives in `Standalone Python Scripts/` and
is **not** a ROS node. It's the one-time tool that produces
`config/FirstCal.json`, the per-whisker pixel-to-millimetre mapping
that `WSKR_range` reads at startup to convert each whisker's hit pixel
into a drive distance.

Flow:
- Place the robot in front of a tape measure or marked reference.
- Snap a frame (or load a previously-saved `.png`).
- Click **Select whisker -30°** (or whichever angle you're calibrating),
  click the matching point on the image, and enter the real-world
  distance in mm. Repeat for several distances per whisker.
- Click **Done** to fit a spline; repeat for all 11 whisker angles.
- Save — this overwrites (or creates) the `FirstCal.json` you'll
  install into `src/WSKR/config/`.

Use this when:
- You changed the camera mounting height or angle.
- Whisker distances are systematically wrong (e.g. the robot thinks
  the wall is 800 mm away but a ruler says 500 mm).
- You want to re-calibrate for a brand-new robot build.

After saving, copy the JSON into `src/WSKR/config/FirstCal.json` and
rebuild `wskr` so the install-share copy is refreshed
(`colcon build --packages-select wskr`).

### Others

- **`Start_Aruco_Approach`** — the demo GUI (covered above).
- **`WSKR_Dashboard`** — the monitoring window (covered above).

---

## The Arduino side

The mecanum base is driven by an Arduino running
`Project5_WS/Arduino/Mecanum_Wheel_Controller_Compact/`. The reference
sketch implements:

- Four-wheel closed-loop speed control (PI + feed-forward) at 20 Hz.
- Full mecanum inverse kinematics (body twist → per-wheel rad/s).
- A CSV serial command protocol at 115200 baud. The commands this
  project cares about are:
  - `V,vx_cm_s,vy_cm_s,omega_deg_s\n` — continuous velocity stream.
  - `X\n` — emergency stop.
- Continuous odometry telemetry emitted after every control tick:
  `O,dyaw_deg,dx_cm,dy_cm,dt_ms\n`.

### Student note — you'll need to update your own sketch

**If your team is starting from a different mecanum sketch (for example,
one that only supports distance-based commands like "drive forward 30 cm
at 15 cm/s"), the ROS side will not be able to talk to it out of the
box.** You must add the `V` streaming command and the `O` odometry
telemetry described above. Without them, `WSKR/cmd_vel` has nowhere to
go and `dead_reckoning_node` can't update when the tag leaves the FOV.

Copy-paste the prompt below into your coding agent of choice (Claude
Code, Cursor, etc.) while you have your existing `.ino` open. It's
written so an agent can do the modification in one pass without removing
anything you already have:

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
| Floor mask empty or noisy | `Floor_Tuner` — `color_distance_threshold`, `val_range`. |
| Whiskers ignore the target when it's inside the frame | They're supposed to — whiskers measure floor, not targets. Target is tracked separately. |
| Heading stays pinned at 0 when tag is visible | Is the approach action actually running? `WSKR/heading_to_target/visual_obs` only publishes during an active goal. |
| Robot drives through the target | Check `proximity_success_mm` and whisker calibration (`config/FirstCal.json`). |
| Robot does nothing after "Start Approach" | Is `mecanum_serial_bridge` connected to the right `port`? Check `ros2 topic echo /odom` — if nothing, the Arduino isn't talking. |
| Mode flips rapidly between visual and dead_reckoning | Hysteresis too tight — check `dr_handoff_deg` / `visual_reacquire_deg` params on `WSKR_dead_reckoning`. |

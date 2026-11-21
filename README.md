# Robot Knife Arm: Cream Cheese on a Bagel

A low-cost robot arm that uses two cameras to find a bagel and a tub of cream cheese, dip a knife
into the tub, and spread it across the bagel. Built at a hackathon in Ithaca by
[@glukeflores](https://github.com/glukeflores) and [@realbenjaminyou](https://github.com/realbenjaminyou).

## The goal

Get a cheap hobby robot arm to do a small, everyday kitchen task — spreading cream cheese on a
bagel — using only what it can see through two ordinary USB cameras. No expensive depth sensors, no
force sensors, no hand-placed coordinates: the arm should work out where things are on its own and
keep working if the bagel, the tub or a camera gets moved.

## Purpose

Spreading is harder than it looks for a robot. The knife has to reach the right spot, get to the
right height above a soft, uneven surface, and cover the whole top without missing patches or
pressing too hard. This project is a working example of solving that with:

- **Vision instead of measuring by hand**: the cameras find the objects and the knife.
- **Self-calibration**: the arm calibrates the cameras by moving around in front of them.
- **Safety first**: every move watches the servo load and stops if the arm hits something
  (the table, an object, a person).

## Hardware

- **Arm:** [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) (6 × Feetech STS3215 servos,
  12 V), with a knife zip-tied to the gripper. The knife blade is marked with green tape so the
  cameras can see it.
- **Cameras:** two fixed USB 2.0 cameras: one looking **down from above** (where things are on the
  table) and one **from the side at table height** (how tall the bagel is, how high the knife is).
- **Computer:** a Mac (camera capture uses a small Swift program built on AVFoundation).
- Two USB hubs plugged straight into the computer: one for the arm, one for both cameras.

## What we built

1. **Arm control with a safety guard** (`robot_up_down.py`, `robot_jog.py`): moves the servos
   smoothly and stops if any joint meets unexpected resistance. It can also find the arm's
   "straight up" pose automatically by sweeping each joint and reading gravity off the servo load.
2. **Arm server** (`arm_server.py`): the one process that talks to the servos. It publishes where
   the arm and knife tip are ~10× a second and carries out moves requested by the other scripts.
3. **Kinematics** (`knife_tasks.py`): turns "put the knife tip here, pointing this way" (in
   centimetres) into joint angles, and back.
4. **Camera capture** (`camera_capture.swift`, `camera_view.py`): streams both cameras at once
   through one hub, which OpenCV can't do on macOS.
5. **Camera calibration** (`camera_calib.py`): uses the arm itself as a moving 3D marker. It drives
   the arm through a set of poses, records where the green tape appears in each camera, and fits
   each camera's projection matrix so 3D points map to pixels.
6. **Scene tracking** (`scene_tracker.py`): finds the bagel and the cream cheese tub in both
   cameras, triangulates their 3D positions, and draws the knife on the video from the arm's joint
   angles.
7. **AI vision** (`live_bagel_robot_vision_v2.py`): a YOLO-World detector plus colour and texture
   checks that spot the bagel, cream cheese, gripper and knife, and estimate how much of the bagel
   is covered.
8. **Bagel surface and coverage** (`bagel_surface.py`): measures the bagel's real curved top from
   the side camera and keeps a 2.5 mm grid of which parts the knife has already covered.
9. **Knife centring** (`knife_center.py`): steers the knife over the tub or the bagel purely by
   what the cameras see. It learns which way each move shows up in each picture from small test
   nudges, so it keeps working if a camera is bumped.
10. **Spreading** (`spread.py`): sweeps the knife back and forth across the bagel in rows, aiming at
    whatever the coverage map says is still bare, until about 90% is covered.
11. **Command center** (`robot_command_center.py`): one window with both live camera views, the
    knife's position and angle, and controls: on-screen joystick and sliders, keyboard, or a game
    controller. It also has one-button actions: knife → tub, knife → bagel, point down, and the full
    tub → bagel sequence.

## What you could do with this code

- **Run the same task** on your own SO-ARM100/101: calibrate your cameras, point it at a bagel, and
  spread.
- **Teach it other spreading or wiping jobs**: butter on toast, frosting on a cake, wiping a
  table. Swap the bagel tracker and coverage map for your own surface.
- **Reuse the camera calibration** (`camera_calib.py`) in any robot project: tape a coloured
  marker to the gripper and let the arm calibrate the cameras itself, with no checkerboard needed.
- **Reuse the safety guard** (`robot_up_down.py`) for any Feetech-servo arm so it stops when it
  bumps into something.
- **Use the two-camera tracking** (`scene_tracker.py`) to get 3D positions of objects on a table
  from two cheap webcams.
- **Drive the arm by hand** with the command center for demos, teleoperation, or collecting
  training data for a learned policy.
- **Learn from it**: forward/inverse kinematics, camera projection and triangulation, and
  closed-loop visual servoing, all in plain Python and NumPy.

## Running it

```bash
pip install opencv-python numpy scipy pyserial ultralytics torch pygame
swiftc -O camera_capture.swift -o camera_capture

python3 robot_up_down.py calibrate     # once: find the arm's straight-up pose
python3 camera_calib.py collect frames # once: calibrate the cameras using the arm
python3 camera_calib.py fit
python3 robot_command_center.py        # live views + controls
python3 spread.py run                  # spread until ~90% of the bagel is covered
```

The serial port, camera IDs and calibration files in this repo are for our setup; change them to
match yours. Each script explains its own options at the top of the file. The YOLO-World weights
download automatically on first run.

> **Safety:** this arm holds a real knife. Keep hands clear while it's powered, and keep the STOP
> button (or Space) within reach.

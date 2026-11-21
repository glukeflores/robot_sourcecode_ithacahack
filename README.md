# Robot Knife Arm: Cream Cheese on a Bagel

We built a robot arm that spreads cream cheese on a bagel. Two USB cameras watch the table, find
the bagel and the tub of cream cheese, and guide a knife held by the arm. Made at a hackathon in
Ithaca by [@glukeflores](https://github.com/glukeflores) and
[@realbenjaminyou](https://github.com/realbenjaminyou).

## Goal

We wanted a cheap hobby robot arm to do a simple kitchen job using only two regular webcams. We
didn't use depth sensors, force sensors, or coordinates typed in by hand. The arm figures out where
things are from the cameras, and it still works if the bagel, the tub, or a camera gets moved.

## Purpose

To spread something, the knife has to get to the right spot, sit at the right height over a soft
and uneven surface, and cover the whole top without skipping spots or pushing too hard. We wanted
to show this can be done with cameras and some math instead of expensive hardware. The arm also
calibrates the cameras by itself, and every move checks the servo load so the arm stops if it runs
into the table, an object, or a person.

## Hardware

- Arm: [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100), six Feetech STS3215 servos at
  12 V. A knife is zip-tied to the gripper, with green tape on the blade so the cameras can see it.
- Cameras: two fixed USB 2.0 cameras. One looks down from above to see where things are on the
  table. The other is at table height on the side to see how tall the bagel is and how high the
  knife is.
- A Mac. The camera capture program is written in Swift.
- Two USB hubs plugged directly into the Mac, one for the arm and one for the cameras.

## What we built

1. Arm control with a safety stop (`robot_up_down.py`, `robot_jog.py`). Moves the servos smoothly
   and stops if a joint hits something. It can also find the arm's straight up position on its own
   by sweeping each joint and reading the servo load.
2. Arm server (`arm_server.py`). The only program that talks to the servos. It reports where the
   arm and knife tip are about 10 times a second and runs moves sent by the other scripts.
3. Kinematics (`knife_tasks.py`). Converts a knife tip position and angle in centimeters into
   joint angles, and the other way around.
4. Camera capture (`camera_capture.swift`, `camera_view.py`). Streams both cameras at the same
   time through one hub, which OpenCV couldn't do on the Mac.
5. Camera calibration (`camera_calib.py`). The arm moves through a set of poses while the cameras
   record where the green tape shows up. From that we fit each camera's projection so we can turn
   3D points into pixels.
6. Scene tracking (`scene_tracker.py`). Finds the bagel and the tub in both cameras, works out
   their 3D positions, and draws the knife on the video using the arm's joint angles.
7. AI vision (`live_bagel_robot_vision_v2.py`). A YOLO-World object detector plus color and
   texture checks to find the bagel, cream cheese, gripper and knife, and to estimate how much of
   the bagel is covered.
8. Bagel surface and coverage (`bagel_surface.py`). Measures the curved top of the bagel from the
   side camera and keeps a 2.5 mm grid of which parts the knife has already touched.
9. Knife centering (`knife_center.py`). Moves the knife over the tub or the bagel using only what
   the cameras see. It makes small test moves first to learn which direction is which in each
   camera, so it still works if a camera gets bumped.
10. Spreading (`spread.py`). Moves the knife back and forth across the bagel in rows, going after
    the spots the coverage map says are still empty, until about 90% is covered.
11. Command center (`robot_command_center.py`). One window with both camera feeds, the knife's
    position and angle, and controls. You can drive it with an on-screen joystick and sliders, the
    keyboard, or a game controller. There are also buttons for knife to tub, knife to bagel, point
    down, and the full tub to bagel sequence.

## What you can do with this code

- Run the same thing on your own SO-ARM100 or SO-ARM101.
- Change it to spread or wipe other things, like butter on toast, frosting on a cake, or cleaning
  a table. You would swap out the bagel tracking and coverage map for your own surface.
- Use the camera calibration in another robot project. Put a colored marker on the gripper and the
  arm calibrates the cameras itself, no checkerboard needed.
- Use the safety stop on any arm with Feetech servos so it stops when it hits something.
- Use the two camera tracking to get 3D positions of objects on a table with cheap webcams.
- Control the arm by hand with the command center, for demos or for recording training data.
- Learn from it. It covers kinematics, camera projection, triangulation and steering a robot with
  camera feedback, all in Python and NumPy.

## Running it

```bash
pip install opencv-python numpy scipy pyserial ultralytics torch pygame
swiftc -O camera_capture.swift -o camera_capture

python3 robot_up_down.py calibrate     # once: find the arm's straight up pose
python3 camera_calib.py collect frames # once: calibrate the cameras using the arm
python3 camera_calib.py fit
python3 robot_command_center.py        # camera views and controls
python3 spread.py run                  # spread until about 90% of the bagel is covered
```

The serial port, camera IDs and calibration files here are for our setup, so change them to match
yours. Each script lists its options at the top of the file. The YOLO-World weights download the
first time you run it.

Safety: the arm is holding a real knife. Keep your hands away while it's powered on and keep the
STOP button (or the space bar) close.

# Kinematics and tasks for the knife arm (SO-ARM101 on ...30541) guided by the overview camera.
#
# Robot frame (cm): origin on the shoulder-lift axis above the base, Z up, X/Y horizontal, with the
# base rotation (pan) measured from servo position PAN0. The "wrist point" is the wrist-flex axis;
# the knife tip sits TOOL_LENGTH cm beyond it along the hand's pointing direction ("pitch",
# degrees from straight up: 90 = horizontal, 180 = straight down).
#
#   python3 knife_tasks.py pose                       # where the wrist and tip are now
#   python3 knife_tasks.py goto X Y Z PITCH [secs]    # move the wrist point (cm) with that hand pitch
#   python3 knife_tasks.py touch PITCH                # lower slowly until the tip touches; record it
#   python3 knife_tasks.py fit                        # tool length, table height, pixel->table mapping
#   python3 knife_tasks.py dip                        # knife straight down into the cream cheese tub
#   python3 knife_tasks.py spread [strokes]           # knife flat, back and forth across the bagel
#
# Every move goes through robot_up_down's contact guard and torque caps. The tip position is
# written to knife_state.json after each move so scene_tracker.py can draw it on the video.
import json
import math
import os
import sys
import time

import numpy as np

import robot_up_down as r

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = "/dev/tty.usbmodem5B901030541"
L1, L2 = 11.3, 13.5          # upper arm and forearm (cm), SO-ARM101
UP = {r.LIFT: 2524, r.ELBOW: 968, r.WRIST: 1869}  # calibrated straight-up pose of this arm
PAN0 = 2048
DEG = 360 / 4096
TOUCHES = os.path.join(HERE, "touches.json")
CALIBRATION = os.path.join(HERE, "calibration.json")
KNIFE_STATE = os.path.join(HERE, "knife_state.json")
TOUCH_STEP = 0.25            # cm per step while feeling for the table
TOUCH_LAG = 0.6              # cm the tip may fall behind its goal before it counts as touching


def load_calibration():
    if os.path.exists(CALIBRATION):
        with open(CALIBRATION) as f:
            return json.load(f)
    return {}


CAL = load_calibration()
TOOL_LENGTH = CAL.get("tool_length")   # cm from wrist point to knife tip, once measured
TABLE_Z = CAL.get("table_z")           # table height in the robot frame, once measured


# ---------------------------------------------------------------- kinematics

def forward(j):
    """Joint positions -> wrist point (X, Y, Z) in cm and hand pitch in degrees."""
    t1 = (j[r.LIFT] - UP[r.LIFT]) * DEG
    t2 = t1 + (j[r.ELBOW] - UP[r.ELBOW]) * DEG
    pitch = t2 + (j[r.WRIST] - UP[r.WRIST]) * DEG
    reach = L1 * math.sin(math.radians(t1)) + L2 * math.sin(math.radians(t2))
    z = L1 * math.cos(math.radians(t1)) + L2 * math.cos(math.radians(t2))
    pan = math.radians((j[r.PAN] - PAN0) * DEG)
    return (reach * math.cos(pan), reach * math.sin(pan), z), pitch


ROLL_REF = 2048  # wrist-roll position that defines the hand frame's zero roll


def hand_axes(j):
    """Unit vectors of the hand frame for joint positions j: u = pointing direction (the way the
    gripper points), v = perpendicular to u in the arm's vertical plane at zero roll, w = sideways.
    Both v and w turn with the wrist roll."""
    pan = math.radians((j[r.PAN] - PAN0) * DEG)
    _, pitch = forward(j)
    p = math.radians(pitch)
    radial = np.array([math.cos(pan), math.sin(pan), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    side = np.array([-math.sin(pan), math.cos(pan), 0.0])
    u = math.sin(p) * radial + math.cos(p) * up
    v = math.cos(p) * radial - math.sin(p) * up
    roll = math.radians((j[r.ROLL] - ROLL_REF) * DEG)
    return u, math.cos(roll) * v + math.sin(roll) * side, -math.sin(roll) * v + math.cos(roll) * side


def hand_point(j, offset):
    """World position (cm) of a point fixed to the gripper at hand-frame `offset` (a, b, c)."""
    wrist, _ = forward(j)
    u, v, w = hand_axes(j)
    return np.array(wrist) + offset[0] * u + offset[1] * v + offset[2] * w


def to_hand(j, point):
    """Inverse of hand_point: hand-frame offset of a world point."""
    wrist, _ = forward(j)
    u, v, w = hand_axes(j)
    d = np.array(point) - np.array(wrist)
    return np.array([d @ u, d @ v, d @ w])


# Servo limits of this arm (from its calibration), used to keep IK answers physically possible.
LIMITS = {r.PAN: (715, 2988), r.LIFT: (1390, 3775), r.ELBOW: (536, 2743), r.WRIST: (825, 3034),
          r.ROLL: (200, 3900)}  # roll kept off its end stops, where it jams
IK_JOINTS = [r.PAN, r.LIFT, r.ELBOW, r.WRIST, r.ROLL]


def knife_points(j):
    """World (cm) positions of the knife's base (at the grip) and tip for joint positions j."""
    tool = load_calibration()["tool"]
    return hand_point(j, tool["knife_base"]), hand_point(j, tool["knife_tip"])


def solve_knife(tip, j0, direction=None, flat=False, fix=(), dir_weight=20.0):
    """Inverse kinematics for the knife itself: joint positions that put the knife tip at `tip`
    (cm) with the blade (base -> tip) pointing along `direction` (unit vector), or lying level if
    `flat`. Starts from j0 (current joints) so the answer stays close to where the arm is.
    Returns (joints, tip error cm, direction error deg)."""
    from scipy.optimize import least_squares

    tip = np.array(tip, float)
    free = [sid for sid in IK_JOINTS if sid not in fix]  # `fix`: joints held at their j0 value
    fixed = {sid: j0[sid] for sid in j0 if sid not in free}

    def joints_of(x):
        return {**fixed, **{sid: float(v) for sid, v in zip(free, x)}}

    def residuals(x):
        base, t = knife_points(joints_of(x))
        blade = (t - base) / np.linalg.norm(t - base)
        res = list(t - tip)
        if direction is not None:
            res += list(dir_weight * (blade - np.array(direction)))
        if flat:
            res.append(20 * blade[2])
        return np.array(res)

    lo = [LIMITS[s][0] for s in free]
    hi = [LIMITS[s][1] for s in free]
    x0 = np.clip([j0[s] for s in free], lo, hi)
    sol = least_squares(residuals, x0, bounds=(lo, hi), x_scale=100.0)
    j = {sid: int(round(v)) for sid, v in joints_of(sol.x).items()}
    base, t = knife_points(j)
    blade = (t - base) / np.linalg.norm(t - base)
    err_deg = 0.0
    if direction is not None:
        err_deg = math.degrees(math.acos(float(np.clip(blade @ np.array(direction), -1, 1))))
    elif flat:
        err_deg = abs(math.degrees(math.asin(float(np.clip(blade[2], -1, 1)))))
    return j, float(np.linalg.norm(t - tip)), err_deg


def tip_of(wrist, pitch, pan_rad, length=None):
    """Knife tip from the wrist point, hand pitch and base angle."""
    length = TOOL_LENGTH if length is None else length
    out = length * math.sin(math.radians(pitch))
    return (wrist[0] + out * math.cos(pan_rad), wrist[1] + out * math.sin(pan_rad),
            wrist[2] + length * math.cos(math.radians(pitch)))


def inverse(x, y, z, pitch):
    """Wrist point (cm) and hand pitch (deg) -> joint positions. Raises ValueError if unreachable."""
    pan = math.atan2(y, x)
    reach = math.hypot(x, y)
    c = (reach * reach + z * z - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1:
        raise ValueError(f"wrist point ({x:.1f}, {y:.1f}, {z:.1f}) is out of reach")
    bend = math.degrees(math.acos(c))
    t1 = math.degrees(math.atan2(reach, z) - math.atan2(L2 * math.sin(math.radians(bend)),
                                                          L1 + L2 * math.cos(math.radians(bend))))
    return {r.PAN: round(PAN0 + math.degrees(pan) / DEG),
            r.LIFT: round(UP[r.LIFT] + t1 / DEG),
            r.ELBOW: round(UP[r.ELBOW] + bend / DEG),
            r.WRIST: round(UP[r.WRIST] + (pitch - t1 - bend) / DEG)}


def wrist_for_tip(tip, pitch, pan_rad):
    """Inverse of tip_of: where the wrist must be for the tip to land on `tip`."""
    out = TOOL_LENGTH * math.sin(math.radians(pitch))
    return (tip[0] - out * math.cos(pan_rad), tip[1] - out * math.sin(pan_rad),
            tip[2] - TOOL_LENGTH * math.cos(math.radians(pitch)))


# ---------------------------------------------------------------- talking to arm_server.py

def arm_cmd(op, timeout=60, **args):
    """Ask arm_server.py to do something; wait for and return its result (raises on failure)."""
    cmd_dir = os.path.join(HERE, "arm_cmds")
    name = f"{time.time():.6f}"
    path = os.path.join(cmd_dir, name + ".cmd.json")
    with open(path + ".tmp", "w") as f:
        json.dump({"op": op, **args}, f)
    os.replace(path + ".tmp", path)
    result_path = os.path.join(cmd_dir, name + ".result.json")
    deadline = time.time() + timeout
    while not os.path.exists(result_path):
        if time.time() > deadline:
            raise TimeoutError(f"arm_server didn't answer '{op}' (is it running?)")
        time.sleep(0.05)
    with open(result_path) as f:
        result = json.load(f)
    os.remove(result_path)
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result["result"]


# ---------------------------------------------------------------- arm helpers

def read_joints(arm):
    return {sid: arm.read_position(sid) for sid in r.ALL_JOINTS}


def publish(arm):
    """Write the current wrist/tip position for the video overlay."""
    j = read_joints(arm)
    wrist, pitch = forward(j)
    pan = math.radians((j[r.PAN] - PAN0) * DEG)
    state = {"time": time.time(), "joints": j, "wrist": [round(v, 2) for v in wrist], "pitch": round(pitch, 1)}
    if TOOL_LENGTH:
        state["tip"] = [round(v, 2) for v in tip_of(wrist, pitch, pan)]
    tmp = KNIFE_STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, KNIFE_STATE)
    return state


def goto(arm, x, y, z, pitch, seconds=2.5):
    target = inverse(x, y, z, pitch)
    r.move_all([arm], [target], seconds)
    time.sleep(0.3)
    return publish(arm)


def touch(arm, pitch):
    """Lower the wrist straight down (hand at `pitch`) until the tip meets the table.

    The servos sag a little under gravity, so the actual height trails the commanded height by a
    roughly steady amount. Once the tip rests on something, the commanded height keeps dropping but
    the actual one doesn't: the gap grows. That growth is the touch.
    """
    j = read_joints(arm)
    (x, y, z), _ = forward(j)
    pan = math.radians((j[r.PAN] - PAN0) * DEG)
    reach = math.hypot(x, y)
    gaps, cmd_z = [], z
    while True:
        cmd_z -= TOUCH_STEP
        r.move_all([arm], [inverse(reach * math.cos(pan), reach * math.sin(pan), cmd_z, pitch)], 0.25)
        time.sleep(0.25)
        (ax, ay, az), apitch = forward(read_joints(arm))
        gaps.append(az - cmd_z)
        base = sum(gaps[:4]) / min(4, len(gaps))
        if len(gaps) > 5 and gaps[-1] - base > TOUCH_LAG and gaps[-2] - base > TOUCH_LAG / 2:
            break
        if cmd_z < z - 30:
            raise RuntimeError("went 30 cm down without touching anything")
    touched = {"wrist": [round(ax, 2), round(ay, 2), round(az, 2)], "pitch": round(apitch, 1),
               "pan": round(pan, 4), "time": time.time()}
    # Back off upward so the tip isn't left pressing on the table.
    r.move_all([arm], [inverse(reach * math.cos(pan), reach * math.sin(pan), az + 2.0, pitch)], 1.0)
    publish(arm)
    return touched


# ---------------------------------------------------------------- calibration

def fit():
    """From the touches: tool length and table height, then the pixel->table homography."""
    with open(TOUCHES) as f:
        touches = json.load(f)
    # Each touch: wrist_z + L*cos(pitch) = table_z. Least squares over (L, table_z).
    A = np.array([[math.cos(math.radians(t["pitch"])), -1.0] for t in touches])
    b = np.array([-t["wrist"][2] for t in touches])
    (length, table_z), *_ = np.linalg.lstsq(A, b, rcond=None)
    residuals = A @ np.array([length, table_z]) - b
    cal = {"tool_length": round(float(length), 2), "table_z": round(float(table_z), 2),
           "touch_residual_cm": round(float(np.abs(residuals).max()), 2)}
    pts = [t for t in touches if t.get("px")]
    if len(pts) >= 4:
        table_xy = np.array([tip_of(t["wrist"], t["pitch"], t["pan"], length)[:2] for t in pts], np.float64)
        pix = np.array([t["px"] for t in pts], np.float64)
        import cv2
        H, _ = cv2.findHomography(pix, table_xy)
        back = cv2.perspectiveTransform(pix.reshape(-1, 1, 2), H).reshape(-1, 2)
        cal["H"] = H.tolist()
        cal["pixel_fit_error_cm"] = round(float(np.linalg.norm(back - table_xy, axis=1).max()), 2)
    with open(CALIBRATION, "w") as f:
        json.dump(cal, f, indent=2)
    return cal


# ---------------------------------------------------------------- tasks

def scene_positions():
    """Latest table positions (cm) of the tub and bagel from scene_tracker.py."""
    with open(os.path.join(HERE, "scene_state.json")) as f:
        s = json.load(f)
    return {k: v["table_cm"] for k, v in s["objects"].items() if v.get("table_cm")}


def main():
    cmd = sys.argv[1]
    arm = r.Arm(PORT)
    try:
        arm.hold()
        if cmd == "pose":
            print(json.dumps(publish(arm)))
        elif cmd == "goto":
            x, y, z, pitch = map(float, sys.argv[2:6])
            secs = float(sys.argv[6]) if len(sys.argv) > 6 else 2.5
            print(json.dumps(goto(arm, x, y, z, pitch, secs)))
        elif cmd == "touch":
            t = touch(arm, float(sys.argv[2]))
            touches = json.load(open(TOUCHES)) if os.path.exists(TOUCHES) else []
            touches.append(t)
            with open(TOUCHES, "w") as f:
                json.dump(touches, f, indent=1)
            print(json.dumps(t))
        elif cmd == "fit":
            print(json.dumps(fit(), indent=1))
        else:
            sys.exit(f"unknown command {cmd}")
    except r.Contact as e:
        arm.freeze()
        print("CONTACT:", e)
    finally:
        arm.close()


if __name__ == "__main__":
    main()

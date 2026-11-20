# Centre the knife, pointing straight down, a set height above an object (the cream cheese tub or
# the bagel). Used by robot_command_center.py, and runnable on its own.
#
#   python3 knife_center.py                         # tip to the centre of the tub
#   python3 knife_center.py --target bagel           # tip to the centre of the bagel
#   python3 knife_center.py --target tub --height 2  # tip 2 cm above the tub rim instead
#
# Needs arm_server.py and camera_capture running (frames in FRAME_DIR). Run alone, object positions
# come from scene_tracker.py's scene_state.json; robot_command_center.py passes its own instead.
# It steers purely by what the cameras see: the knife's green-taped blade versus the object, in both
# views. Which way a move shows up in each picture is learned from small test nudges, so it keeps
# working if a camera is bumped.
#
#   from knife_center import center_knife_over
#   center_knife_over("tub", height_cm=1.0)
import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

import knife_tasks as k
import robot_up_down as r

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = os.environ.get(
    "FRAME_DIR", os.path.join(HERE, "frames"))
TOP, SIDE = "0x1110000328f0073", "0x1130000328f0073"
TAPE_HSV = ((40, 60, 40), (105, 255, 255))  # green tape; the side camera sees it bluer (hue ~100)
MAX_STEP_CM = 3.0
CENTRED_PX = 8           # close enough in each view
WORK_Z_CM = 0.0          # start centring no higher than this (arm z; the table is at about -10)
TIP_TOLERANCE_CM = 0.4   # how far the tip may miss a commanded point before the blade is let lean
SURFACE_OBJECTS = {"bagel"}  # objects aimed at their real top surface (side camera), not box centre
SURFACE_PRESS_CM = 0.3   # ...and pressed this far into it, so the blade actually touches
MIN_SCORE = 0.6


class CenteringError(RuntimeError):
    pass


# ---------------------------------------------------------------- seeing

def knife_tip_px(frame):
    """Tip of the green-taped blade: the long thin green patch's end farthest from the gripper tape."""
    if frame is None:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(cv2.inRange(hsv, *TAPE_HSV), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 60 or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        t = (pts - [cx, cy]) @ np.array([vx, vy])
        length = float(t.max() - t.min())
        ends = [np.array([cx, cy]) + np.array([vx, vy]) * t.min(), np.array([cx, cy]) + np.array([vx, vy]) * t.max()]
        blobs.append({"length": length, "thin": area / max(length, 1) <= length / 3, "ends": ends,
                      "centre": centroids[i], "area": area})
    blades = [b for b in blobs if b["thin"] and b["length"] >= 40]
    if not blades:
        return None
    blade = max(blades, key=lambda b: b["length"])
    others = [b for b in blobs if b is not blade]
    if others:  # gripper tape present: tip is the far end from it
        grip = max(others, key=lambda b: b["area"])["centre"]
        return max(blade["ends"], key=lambda e: np.linalg.norm(e - grip))
    return max(blade["ends"], key=lambda e: e[1])  # otherwise the lower end


def boxes_from_scene_state(name, cam):
    """Default object locator: scene_tracker.py's latest box for `name` in camera `cam`."""
    with open(os.path.join(HERE, "scene_state.json")) as f:
        obj = json.load(f)["cameras"][cam]["objects"][name]
    return obj["box"], obj["score"]


def object_px(name, cam, locate):
    """Object centre and top edge (pixels) in a camera (centre of its tracked box)."""
    found = locate(name, cam)
    if not found:
        raise CenteringError(f"{name} not found in camera {cam}")
    (x0, y0, x1, y1), score = found
    if score < MIN_SCORE:
        raise CenteringError(f"{name} not found in camera {cam} (match {score:.2f}); re-identify it first")
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2]), float(y0)


def surface_px(name, column, locate, fallback):
    """Height (pixel row) of an object's real top surface in the side view at pixel `column`,
    from its traced top edge (bagel_surface.side_top_curve); the box top if it can't be traced."""
    import bagel_surface
    box, _ = locate(name, SIDE)
    curve = bagel_surface.side_top_curve(read_frame(SIDE), box)
    if not curve:
        return fallback
    us, vs = np.array([c[0] for c in curve]), np.array([c[1] for c in curve])
    return float(np.interp(column, us, vs))


def read_frame(cam):
    return cv2.imread(os.path.join(FRAME_DIR, f"live_{cam}.jpg"))


def look(stop, retries=6):
    """Knife tip pixels in (top, side), retrying past frames that miss the tape."""
    for _ in range(retries):
        if stop.is_set():
            raise CenteringError("stopped")
        time.sleep(0.45)  # a frame taken after the arm has stopped (frames refresh every 0.2 s)
        tips = [knife_tip_px(read_frame(cam)) for cam in (TOP, SIDE)]
        if all(t is not None for t in tips):
            return tips
    raise CenteringError("lost the knife's green tape in a camera")


# ---------------------------------------------------------------- moving

def move_tip(target, joints, seconds=0.5):
    """Put the knife tip at `target` (cm) with the blade as close to straight down as reachable.
    The tip position comes first: far out (the bagel is past the ~24 cm where the blade can still
    point straight down) the blade is allowed to lean rather than the tip missing its target."""
    sol, err, tilt = k.solve_knife(target, joints, direction=[0, 0, -1], dir_weight=3.0)
    if err > TIP_TOLERANCE_CM:  # vertical isn't reachable here: let the blade lean
        sol, err, tilt = k.solve_knife(target, joints, direction=[0, 0, -1], dir_weight=0.3)
    if err > TIP_TOLERANCE_CM:
        raise CenteringError(f"the knife tip can't reach {np.round(target, 1)} ({err:.1f} cm short)")
    state = k.arm_cmd("joints", joints={str(s): v for s, v in sol.items()}, seconds=seconds)
    return {int(a): b for a, b in state["joints"].items()}, tilt


def center_knife_over(target="tub", height_cm=1.0, log=print, locate=boxes_from_scene_state, stop=None,
                      below_cm=0.0):
    """Centre the knife tip over `target` ("tub" or "bagel"), blade pointing down. With
    `height_cm` a number, the tip ends that far above the object's top edge (side camera); with
    `height_cm=None` it goes all the way to the centre of the object's box in both cameras, then
    a further `below_cm` lower.
    `locate(name, cam)` returns (box, score); setting `stop` (a threading.Event) halts it between moves.
    Returns the final tip target (arm coordinates, cm) and the blade's tilt from vertical (deg)."""
    stop = stop or threading.Event()
    state = k.arm_cmd("state")
    joints = {int(a): b for a, b in state["joints"].items()}
    tip = np.array(state["points"]["knife_tip"], float)
    goal_top, _ = object_px(target, TOP, locate)
    goal_side, top_edge_side = object_px(target, SIDE, locate)

    # 1. Blade straight down where it is, so the tip is what both cameras track from here on - and
    #    no higher than the working height, so the taped blade is inside both pictures.
    tip[2] = min(tip[2], WORK_Z_CM)
    joints, tilt = move_tip(tip, joints, 1.0)

    # 2. Learn how 1 cm moves in x, y, z show up in the pictures (top u, top v, side u, side v).
    def reading():
        t, s = look(stop)
        return np.array([t[0], t[1], s[0], s[1]])

    base = reading()
    J = np.zeros((4, 3))
    for i, d in enumerate(np.eye(3)):
        joints, _ = move_tip(tip + d, joints)
        J[:, i] = reading() - base
        joints, _ = move_tip(tip, joints)
        base = reading()
    log("pixels per cm (rows: top u, top v, side u, side v; cols: x, y, z):\n" + str(J.round(1)))
    side_px_per_cm = abs(J[3, 2])
    if side_px_per_cm < 2:
        raise CenteringError("couldn't see the knife move up and down in the side view")

    # 3. Centre over the object, keeping the height. Only the left-right position in each picture
    #    is used: the top camera looks down at an angle, so its up-down position of a hovering tip
    #    depends on how high it is, and only lines up with the object once the tip is down at it.
    rows = [0, 2]  # top u, side u

    def centre_error():
        cur = reading()
        return cur, np.array([goal_top[0] - cur[0], goal_side[0] - cur[2]])

    for it in range(12):
        cur, err = centre_error()
        log(f"centring over {target} {it}: off top {err[0]:.0f} px, side {err[1]:.0f} px")
        if np.abs(err).max() < CENTRED_PX:
            break
        step = np.linalg.lstsq(J[rows, :2], err, rcond=None)[0]
        step *= min(1.0, MAX_STEP_CM / max(np.linalg.norm(step), 1e-6))
        tip = tip + [step[0], step[1], 0.0]
        joints, tilt = move_tip(tip, joints)
    else:
        raise CenteringError(f"couldn't centre over the {target} in 12 moves")

    # 4. Go up or down until the side view shows the tip at the goal height: `height_cm` above the
    #    object's top edge, or (height_cm=None) level with the middle of its box - except for the
    #    bagel, whose box takes in the plate below it: there the goal is its real top surface.
    if target in SURFACE_OBJECTS:
        surface_v = surface_px(target, goal_side[0], locate, top_edge_side)
        log(f"{target} surface in the side view at y={surface_v:.0f} (box top {top_edge_side:.0f})")
        goal_v = surface_v + (SURFACE_PRESS_CM + below_cm) * side_px_per_cm if height_cm is None \
            else surface_v - height_cm * side_px_per_cm
    else:
        goal_v = goal_side[1] + below_cm * side_px_per_cm if height_cm is None \
            else top_edge_side - height_cm * side_px_per_cm
    for it in range(15):
        cur, err = centre_error()
        # Image rows grow downward: a tip drawn above the goal row is physically above it.
        above_cm = (goal_v - cur[3]) / side_px_per_cm  # + = tip still above the goal
        log(f"height {it}: tip {above_cm:+.1f} cm above the goal, off centre top {err[0]:.0f} side {err[1]:.0f} px")
        if abs(above_cm) < 0.25 and np.abs(err).max() < 2 * CENTRED_PX:
            break
        dz = float(np.clip(-above_cm, -1.5, 1.5))  # at most 1.5 cm per move: down when above
        # Keep it centred on the way down: the arm drifts a little as it bends.
        dxy = np.linalg.lstsq(J[rows, :2], err, rcond=None)[0] if np.abs(err).max() >= CENTRED_PX else np.zeros(2)
        dxy *= min(1.0, 1.0 / max(np.linalg.norm(dxy), 1e-6))
        tip = tip + [dxy[0], dxy[1], dz]
        joints, tilt = move_tip(tip, joints, 0.25 if dz > 0 else 0.4)  # up quicker: gravity drags
    where = (f"{below_cm} cm below the centre of" if below_cm else "at the centre of") if height_cm is None \
        else f"{height_cm} cm above"
    log(f"done: tip {where} the {target}, blade {tilt:.1f} deg from vertical")
    return tip, tilt


def make_knife_flat(seconds=0.8, log=print):
    """Turn the blade level (horizontal). The wrist roll is held still (turning it to level the
    blade drives it into its end stop). With the knife mounted as it is, a level blade needs the
    hand pointing out horizontally, so low down or close in the tip has to move: this finds the
    nearest spot that works (up to 14 cm higher, 4 cm in to 8 cm out), lifts the knife there
    first and then levels it, so the big swing happens up in the air. Returns the blade's angle
    from horizontal (degrees)."""
    state = k.arm_cmd("state")
    joints = {int(a): b for a, b in state["joints"].items()}
    tip = np.array(state["points"]["knife_tip"], float)
    radial = np.array([tip[0], tip[1], 0.0]) / max(np.hypot(tip[0], tip[1]), 1e-6)
    shifts = sorted(((up, out) for up in range(0, 15) for out in (0, 2, -2, 4, -4, 6, 8)),
                    key=lambda m: np.hypot(*m))  # nearest first
    for up, out in shifts:
        goal = tip + [0, 0, up] + radial * out
        sol, err, angle = k.solve_knife(goal, joints, flat=True, fix=(r.ROLL,))
        if err <= TIP_TOLERANCE_CM and angle <= 3:
            break
    else:
        raise CenteringError("can't level the blade anywhere near here")
    if up or out:
        log(f"moving the tip {up} cm up and {out:+d} cm outward so the blade can lie flat")
        lift, _, _ = k.solve_knife(tip + [0, 0, up], joints, direction=_blade_direction(state), dir_weight=3.0,
                                   fix=(r.ROLL,))
        joints = {int(a): b for a, b in k.arm_cmd("joints", joints={str(s): v for s, v in lift.items()},
                                                   seconds=seconds)["joints"].items()}
        sol, err, angle = k.solve_knife(goal, joints, flat=True, fix=(r.ROLL,))
    k.arm_cmd("joints", joints={str(s): v for s, v in sol.items()}, seconds=seconds)
    log(f"blade flat ({angle:.1f} deg from horizontal)")
    return angle


def _blade_direction(state):
    base, tip = (np.array(state["points"][p], float) for p in ("knife_base", "knife_tip"))
    return (tip - base) / np.linalg.norm(tip - base)


def center_knife_over_tub(height_cm=1.0, **kw):
    return center_knife_over("tub", height_cm, **kw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centre the knife over the tub or bagel.")
    parser.add_argument("--target", choices=("tub", "bagel"), default="tub")
    parser.add_argument("--height", type=float, default=None,
                        help="cm above the object's top (default: tip to the centre of the object)")
    args = parser.parse_args()
    try:
        center_knife_over(args.target, args.height)
    except (CenteringError, RuntimeError) as e:
        sys.exit(f"stopped: {e}")

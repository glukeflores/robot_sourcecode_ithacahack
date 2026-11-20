# Calibrates both overview cameras and measures the knife, using the arm as a moving 3D marker.
#
#   python3 camera_calib.py collect FRAME_DIR   # drive the arm through poses, record tape pixels
#   python3 camera_calib.py fit                 # camera matrices + tape offset (writes calibration.json)
#   python3 camera_calib.py knife               # knife base/tip from marked pixels -> tool points, length
#
# The green tape on the gripper is a point fixed to the hand at an unknown offset. For each camera,
# the 3x4 projection matrix and that offset are fitted together so the tape's predicted pixels
# (from the joint angles) match where it was seen. Needs arm_server.py running.
import json
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

import knife_tasks as k
from scene_tracker import find_gripper

HERE = k.HERE
SAMPLES = os.path.join(HERE, "calib_samples.json")
SAMPLE_DIR = os.path.join(HERE, "calib_frames")
CAMERAS = {"top": "0x1110000328f0073", "side": "0x1130000328f0073"}
SIDE_IGNORE_LEFT = 60  # px


def poses():
    """Wrist poses (x, y, z, pitch) spread over the area in front of the arm, knife kept high."""
    out = []
    for pan in (-30, -10, 10, 30):
        for radius, z, pitch in ((12, 15, 110), (16, 10, 105), (12, 9, 125), (16, 14, 120)):
            a = math.radians(pan)
            out.append((radius * math.cos(a), radius * math.sin(a), z, pitch))
    return out


MIN_CLEARANCE = 4.0  # cm the knife must stay above the table, bagel and tub during calibration


OBSTACLES = {"bagel_top": 12.0, "tub_top": 7.0}  # tracked object -> radius (cm) it guards
OBJECT_HEIGHT = 4.5  # cm: the tub/bagel tops sit about this far above the table
CLEAR_TABLE_Z = k.load_calibration().get("table_z", -8.0)  # cm: measured by touching the knife tip to the table


def clearance(j, world):
    """Lowest height (cm) of the knife's base, middle or tip above whatever is under it
    (the table, and any tracked bagel or tub that's actually on it)."""
    present = {name: np.array(world[name]) for name in OBSTACLES if world.get(name)}
    table = min(p[2] for p in present.values()) - OBJECT_HEIGHT if present else CLEAR_TABLE_Z
    worst = 99.0
    base, tip = k.knife_points(j)
    for p in (tip, (tip + base) / 2, base):
        h = p[2] - table
        for name, top in present.items():
            if np.hypot(*(p[:2] - top[:2])) < OBSTACLES[name]:
                h = min(h, p[2] - top[2])
        worst = min(worst, h)
    return worst


def safe_pose(pose, prev, world):
    """Raise a pose until it and the move into it keep the knife MIN_CLEARANCE above everything."""
    x, y, z, pitch = pose
    # The move only has to stay as high as it starts (the arm may sag a little below the line
    # after a low pose, and that start can't be raised), or MIN_CLEARANCE, whichever is lower.
    need = min(MIN_CLEARANCE, clearance(prev, world) - 0.5)
    for _ in range(8):
        j = k.inverse(x, y, z, pitch)
        j[5], j[6] = k.ROLL_REF, prev[6]
        if clearance(j, world) < MIN_CLEARANCE:
            z += 2.0
            continue
        path = min(clearance({s: prev[s] + (j[s] - prev[s]) * f for s in j}, world) for f in np.linspace(0, 1, 30))
        if path >= need:
            return (x, y, z, pitch)
        z += 2.0
    raise RuntimeError("no safe height for this pose")


def read_world(clear_table):
    """Obstacles to keep the knife away from: the tracked bagel/tub, or nothing on a cleared table."""
    if clear_table:
        return {}
    with open(os.path.join(HERE, "scene_state.json")) as f:
        world = json.load(f)["world_cm"]
    if not any(world.get(name) for name in OBSTACLES):
        sys.exit("scene_tracker.py must see the tub or bagel (or pass --clear-table with the table empty)")
    return world


def collect(frame_dir, clear_table=False):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    world = read_world(clear_table)
    prev = {int(a): b for a, b in k.arm_cmd("state")["joints"].items()}
    samples = []
    for n, pose in enumerate(poses()):
        # Re-read the scene before every move: things get put on the table mid-run.
        world = read_world(clear_table)
        try:
            x, y, z, pitch = safe_pose(pose, prev, world)
            if z != pose[2]:
                print(f"pose {n}: raised from z {pose[2]} to {z} cm to keep the knife clear")
            state = k.arm_cmd("goto", x=x, y=y, z=z, pitch=pitch, roll=k.ROLL_REF, seconds=2.5)
        except (RuntimeError, ValueError) as e:
            print(f"pose {n}: skipped ({e})")
            continue
        prev = {int(a): b for a, b in state["joints"].items()}
        time.sleep(1.0)  # let both cameras deliver a frame taken after the arm stopped
        sample = {"joints": state["joints"], "px": {}}
        for name, cam in CAMERAS.items():
            frame = cv2.imread(os.path.join(frame_dir, f"live_{cam}.jpg"))
            keep = os.path.join(SAMPLE_DIR, f"{n:02d}_{name}.jpg")
            cv2.imwrite(keep, frame)
            grip = find_gripper(frame)
            if grip and name == "side" and grip[0] < SIDE_IGNORE_LEFT:
                grip = None  # a teal box sits at the side view's left edge
            if grip:
                sample["px"][name] = [round(grip[0], 1), round(grip[1], 1)]
        samples.append(sample)
        print(f"pose {n}: tape seen in {sorted(sample['px'])}", flush=True)
    with open(SAMPLES, "w") as f:
        json.dump(samples, f, indent=1)


def dlt(points3d, pixels):
    """Direct linear transform: 3x4 projection matrix from >= 6 point correspondences."""
    rows = []
    for (X, Y, Z), (u, v) in zip(points3d, pixels):
        rows.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        rows.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    _, _, vt = np.linalg.svd(np.array(rows))
    return vt[-1].reshape(3, 4)


def project(P, point):
    p = P @ np.append(point, 1.0)
    return p[:2] / p[2]


MIN_SAMPLES = 9  # per camera; fewer and the 11-number camera model just memorises the points


def fit():
    with open(SAMPLES) as f:
        samples = json.load(f)
    for name in CAMERAS:
        n = sum(name in s["px"] for s in samples)
        if n < MIN_SAMPLES:
            sys.exit(f"only {n} {name}-view samples (need {MIN_SAMPLES}); keeping the existing calibration")
    for s in samples:
        s["joints"] = {int(a): b for a, b in s["joints"].items()}

    def residuals(offset):
        res = []
        for name in CAMERAS:
            used = [s for s in samples if name in s["px"]]
            pts = [k.hand_point(s["joints"], offset) for s in used]
            P = dlt(pts, [s["px"][name] for s in used])
            res += [c for s, p in zip(used, pts) for c in project(P, p) - s["px"][name]]
        return np.array(res)

    # Soft-L1 loss so one bad detection can't drag the whole fit.
    best = least_squares(residuals, x0=np.array([10.0, 0.0, 0.0]), loss="soft_l1", f_scale=10.0)
    offset = best.x
    cal = k.load_calibration()
    cal.setdefault("tool", {})["tape"] = [round(float(v), 2) for v in offset]
    cal["cameras"] = {}
    for name, cam in CAMERAS.items():
        used = [s for s in samples if name in s["px"]]
        pts = [k.hand_point(s["joints"], offset) for s in used]
        P = dlt(pts, [s["px"][name] for s in used])
        err = [float(np.linalg.norm(project(P, p) - s["px"][name])) for s, p in zip(used, pts)]
        cal["cameras"][cam] = {"name": name, "P": (P / np.linalg.norm(P[2, :3])).tolist(),
                               "samples": len(used), "rms_px": round(float(np.sqrt(np.mean(np.square(err)))), 2),
                               "max_px": round(max(err), 2)}
    with open(k.CALIBRATION, "w") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps({"tape_offset_cm": cal["tool"]["tape"],
                      **{c["name"]: {x: c[x] for x in ("samples", "rms_px", "max_px")} for c in cal["cameras"].values()}},
                     indent=1))


def triangulate(pixels):
    """3D point from its pixel in each calibrated camera: {camera id: (u, v)}."""
    cal = k.load_calibration()
    rows = []
    for cam, (u, v) in pixels.items():
        P = np.array(cal["cameras"][cam]["P"])
        rows += [u * P[2] - P[0], v * P[2] - P[1]]
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    return X[:3] / X[3]


def knife():
    """Knife base and tip, as fixed points in the gripper's frame, fitted to all the marked pixels
    (knife_marks.json) in both cameras at once. Fitting to pixels, rather than triangulating each
    pose and averaging, lets each camera count where it sees well: from above the blade often
    points into the lens, so the top view says little about how far along it the tip is."""
    with open(SAMPLES) as f:
        samples = json.load(f)
    with open(os.path.join(HERE, "knife_marks.json")) as f:
        marks = json.load(f)  # {"pose": {"base": {"top": [u, v], "side": [u, v]}, "tip": {...}, ["joints"]}}
    cal = k.load_calibration()
    P = {view: np.array(cal["cameras"][cam]["P"]) for view, cam in CAMERAS.items()}
    for point in ("base", "tip"):
        obs = []
        for idx, m in marks.items():
            if point not in m:
                continue
            src = m["joints"] if "joints" in m else samples[int(idx)]["joints"]
            j = {int(a): b for a, b in src.items()}
            obs += [(j, view, np.array(px, float)) for view, px in m[point].items()]

        def residuals(off):
            return np.concatenate([project(P[view], k.hand_point(j, off)) - px for j, view, px in obs])

        x0 = np.array(cal["tool"].get("knife_" + point, [20.0, 0.0, 0.0]))
        fit_ = least_squares(residuals, x0, loss="soft_l1", f_scale=8.0)
        err = np.abs(fit_.fun).reshape(-1, 2)
        cal["tool"]["knife_" + point] = [round(float(v), 2) for v in fit_.x]
        print(f"knife {point}: hand offset {fit_.x.round(2)} cm; pixel error mean {np.linalg.norm(err, axis=1).mean():.1f}, "
              f"max {np.linalg.norm(err, axis=1).max():.1f} over {len(obs)} marks")
    length = float(np.linalg.norm(np.array(cal["tool"]["knife_tip"]) - np.array(cal["tool"]["knife_base"])))
    cal["knife_length"] = round(length, 1)
    with open(k.CALIBRATION, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"knife length (base to tip): {length:.1f} cm")


def knife_auto():
    """Mark the knife's base and tip in every calibration frame automatically: predict the blade
    from the current knife model, snap it onto the black blade in each view, and keep poses where
    both views found it. Then re-measure the knife from those marks."""
    from scene_tracker import project, snap_knife
    with open(SAMPLES) as f:
        samples = json.load(f)
    cal = k.load_calibration()
    marks = {}
    for n, smp in enumerate(samples):
        j = {int(a): b for a, b in smp["joints"].items()}
        base, tip = k.knife_points(j)
        found = {"base": {}, "tip": {}}
        for view, cam in CAMERAS.items():
            frame = cv2.imread(os.path.join(SAMPLE_DIR, f"{n:02d}_{view}.jpg"))
            P = cal["cameras"][cam]["P"]
            b, t = project(P, base), project(P, tip)
            if not all(0 <= v[0] < 640 and 0 <= v[1] < 480 for v in (b, t)):
                continue
            b, t, ok = snap_knife(frame, b, t)
            if ok:
                found["base"][view], found["tip"][view] = b, t
        if len(found["tip"]) == 2:
            marks[str(n)] = found
    with open(os.path.join(HERE, "knife_marks.json"), "w") as f:
        json.dump(marks, f, indent=1)
    print(f"blade found in both views at {len(marks)} poses: {sorted(marks, key=int)}")
    knife()


if __name__ == "__main__":
    {"collect": lambda: collect(sys.argv[2], "--clear-table" in sys.argv), "fit": fit, "knife": knife, "knife_auto": knife_auto}[sys.argv[1]]()

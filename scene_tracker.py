# Live 3D view of the scene from both overview cameras: the bagel, the cream cheese tub and the
# whole knife, with positions and distances in centimetres. Shows the annotated cameras side by side
# and writes everything to scene_state.json each frame.
#
#   python3 scene_tracker.py FRAME_DIR             # live window; q to quit
#   python3 scene_tracker.py FRAME_DIR --once      # process the latest frames, print JSON, exit
#
# Frames come from camera_capture (live_<camera id>.jpg in FRAME_DIR).
# - Bagel and tub: template matching around their last position in each camera (templates and
#   reference boxes in scene_refs.json). Each has an "anchor" pixel (the top centre) that moves with
#   its box; the two cameras' anchors are triangulated into a 3D point.
# - Knife: from the arm's joint angles (knife_state.json, kept fresh by arm_server.py) and the
#   measured knife points in the gripper (calibration.json "tool"), projected into each camera. The
#   blade looks just like the black cables to a colour filter, so it isn't searched for in the image.
import json
import os
import sys
import time

import cv2
import numpy as np

import bagel_surface as bs

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "scene_refs.json")
CALIBRATION = os.path.join(HERE, "calibration.json")
KNIFE_STATE = os.path.join(HERE, "knife_state.json")
STATE = os.path.join(HERE, "scene_state.json")
SEARCH = 60       # pixels around the last position to search for a template
DIP_REACH = 22.0  # cm from the shoulder axis: inside this the knife can dip straight down into the tub
SPREAD_ZONE = (14.0, 22.0)  # cm: where the near-vertical blade can reach bagel height
MIN_SCORE = 0.45  # weaker matches keep the last known position
TOUCH_PX = 2       # side view: blade tip within this many px of the bagel edge counts as touching
JUMP_SCORE = 0.75  # a match elsewhere in the frame must be at least this good to move there
COLORS = {"bagel": (0, 200, 255), "tub": (255, 200, 0), "knife": (0, 0, 255)}


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def project(P, xyz):
    p = np.array(P) @ np.append(np.array(xyz, float), 1.0)
    return [int(round(p[0] / p[2])), int(round(p[1] / p[2]))]


def triangulate(views):
    """3D point from [(P, (u, v)), ...] in two or more calibrated cameras."""
    rows = []
    for P, (u, v) in views:
        P = np.array(P)
        rows += [u * P[2] - P[0], v * P[2] - P[1]]
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    return X[:3] / X[3]


def find_gripper(frame):
    """Centroid of the gripper's green tape: the biggest green patch not touching the frame edge."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (60, 70, 50), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(green)
    h, w = green.shape
    inside = [i for i in range(1, n)
              if stats[i, cv2.CC_STAT_LEFT] > 0 and stats[i, cv2.CC_STAT_TOP] > 0
              and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] < w
              and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] < h
              and stats[i, cv2.CC_STAT_AREA] >= 30]
    if not inside:
        return None
    biggest = max(inside, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return float(centroids[biggest][0]), float(centroids[biggest][1])


SNAP_WIDTH = 24     # px either side of the predicted blade line to look for the real (black) blade
SNAP_MIN_PIXELS = 40


def snap_knife(frame, base, tip, width=None):
    """Move the predicted knife line onto the actual black blade seen in the frame.

    The arm's kinematics put the blade within a couple of centimetres; here the dark pixels in a
    strip around that prediction are fitted with a line, and the tip is where the dark run ends.
    Returns (base_px, tip_px, snapped?)."""
    b, t = np.array(base, float), np.array(tip, float)
    axis = t - b
    length = np.linalg.norm(axis)
    if length < 5:
        return base, tip, False
    axis /= length
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))
    ys, xs = np.nonzero(dark)
    rel = np.stack([xs - b[0], ys - b[1]], axis=1)
    along = rel @ axis
    across = np.abs(rel @ np.array([-axis[1], axis[0]]))
    keep = (along > -0.1 * length) & (along < 2.0 * length) & (across < (width or SNAP_WIDTH))
    if keep.sum() < SNAP_MIN_PIXELS:
        return base, tip, False
    pts = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    d = np.array([vx, vy])
    if d @ axis < 0:
        d = -d
    p0 = np.array([x0, y0])
    # Keep only pixels close to the fitted line (drops cable crossings), then find where the blade ends.
    rel2 = pts - p0
    close = np.abs(rel2 @ np.array([-d[1], d[0]])) < 6
    if close.sum() < SNAP_MIN_PIXELS // 2:
        return base, tip, False
    t_along = np.sort(rel2[close] @ d)
    # The blade is a continuous dark run: stop at the first gap of > 12 px after the base.
    start = float((b - p0) @ d)
    run = t_along[t_along >= start - 5]
    end = run[0] if len(run) else start
    for v in run:
        if v - end > 12:
            break
        end = v
    new_base = p0 + d * start
    new_tip = p0 + d * end
    if np.linalg.norm(new_tip - new_base) < 0.4 * length:  # implausibly short: don't trust it
        return base, tip, False
    return [int(round(v)) for v in new_base], [int(round(v)) for v in new_tip], True


def find_blade(frame, anchor, near=90):
    """The knife blade as a dark, long, thin blob next to `anchor` (the gripper's tape, or where the
    arm model puts the knife's base when the tape is hidden). Returns (base_px, tip_px) or None:
    base = the blade end nearest the anchor, tip = the far end."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 0), (180, 255, 70)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    h, w = dark.shape
    a = np.array(anchor, float)
    best, best_score = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 150 or x == 0 or y == 0 or x + bw >= w or y + bh >= h:  # specks, and cables running off-frame
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        d = np.linalg.norm(pts - a, axis=1)
        if d.min() > near:
            continue
        vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        axis = np.array([vx, vy])
        t = (pts - [cx, cy]) @ axis
        length = t.max() - t.min()
        width = area / max(length, 1)
        if length < 50 or width > length / 3:  # must be long and thin
            continue
        ends = [np.array([cx, cy]) + axis * t.min(), np.array([cx, cy]) + axis * t.max()]
        ends.sort(key=lambda e: e[1])  # the knife hangs down from the gripper: tip is the lower end
        if ends[1][1] - ends[0][1] < 0.4 * length:  # mostly sideways: a cable, not the blade
            continue
        score = length / (1 + d.min() / 30)
        if score > best_score:
            best, best_score = ends, score
    if best is None:
        return None
    return [int(round(v)) for v in best[0]], [int(round(v)) for v in best[1]]


def green_blobs(frame):
    """Green-tape patches not touching the frame edge: [(pixels Nx2, length, width, centre, axis)]."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.morphologyEx(cv2.inRange(hsv, (40, 60, 40), (95, 255, 255)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(green)
    h, w = green.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 60 or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        axis = np.array([vx, vy])
        t = (pts - [cx, cy]) @ axis
        length = float(t.max() - t.min())
        out.append({"pts": pts, "length": length, "width": area / max(length, 1), "centre": np.array([cx, cy]),
                    "axis": axis, "ends": [np.array([cx, cy]) + axis * t.min(), np.array([cx, cy]) + axis * t.max()]})
    return out


def find_knife_tape(frame):
    """The green-taped blade: the longest thin green patch. Base = its end nearest the gripper's
    (compact) green patch, or its upper end if that's hidden; tip = the other end.
    Returns (base_px, tip_px, gripper_px or None) or None."""
    blobs = green_blobs(frame)
    thin = [b for b in blobs if b["length"] >= 50 and b["width"] <= b["length"] / 3]
    if not thin:
        return None
    blade = max(thin, key=lambda b: b["length"])
    others = [b for b in blobs if b is not blade]
    grip = max(others, key=lambda b: len(b["pts"]))["centre"] if others else None
    ends = blade["ends"]
    if grip is not None:
        ends = sorted(ends, key=lambda e: np.linalg.norm(e - grip))
    else:
        ends = sorted(ends, key=lambda e: e[1])
    to_px = lambda e: [int(round(v)) for v in e]
    return to_px(ends[0]), to_px(ends[1]), (to_px(grip) if grip is not None else None)


class CameraTracker:
    """Template tracking of the bagel and tub in one camera."""

    def __init__(self, cam, ref, calibration):
        frame = cv2.imread(os.path.join(HERE, ref["frame"]))
        self.cam, self.label = cam, ref.get("label", cam)
        self.top_line = ref.get("top_line", False)
        self.snap = ref.get("snap_knife", False)
        self.snap_width = ref.get("snap_width", SNAP_WIDTH)
        self.templates, self.boxes, self.scores = {}, {}, {}
        self.ref_boxes, self.anchors = {}, ref.get("anchors", {})
        for name, (x0, y0, x1, y1) in ref["boxes"].items():
            self.templates[name] = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            self.boxes[name] = self.ref_boxes[name] = (x0, y0, x1, y1)
        self.P = calibration.get("cameras", {}).get(cam, {}).get("P")
        self.role = ref.get("role")
        # Bagel-only mask in the reference frame (top view), moved along with the tracked box.
        self.bagel_mask_ref = bs.bagel_mask_from_box(frame, ref["boxes"]["bagel"]) if self.role == "top" else None

    def bagel_mask(self):
        dx = self.boxes["bagel"][0] - self.ref_boxes["bagel"][0]
        dy = self.boxes["bagel"][1] - self.ref_boxes["bagel"][1]
        h, w = self.bagel_mask_ref.shape
        return cv2.warpAffine(self.bagel_mask_ref, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))

    def _match(self, gray, name):
        tpl = self.templates[name]
        th, tw = tpl.shape
        x0, y0, _, _ = self.boxes[name]
        sx0, sy0 = max(0, x0 - SEARCH), max(0, y0 - SEARCH)
        sx1, sy1 = min(gray.shape[1], x0 + tw + SEARCH), min(gray.shape[0], y0 + th + SEARCH)
        window = gray[sy0:sy1, sx0:sx1]
        if window.shape[0] < th or window.shape[1] < tw:
            return
        res = cv2.matchTemplate(window, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        if score < MIN_SCORE:  # moved further than the search window: look over the whole frame
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, far_score, _, (fx, fy) = cv2.minMaxLoc(res)
            if far_score >= JUMP_SCORE:  # only jump on a convincing match, never onto lookalikes
                score, mx, my, sx0, sy0 = far_score, fx, fy, 0, 0
        self.scores[name] = score
        if score >= MIN_SCORE:
            self.boxes[name] = (sx0 + mx, sy0 + my, sx0 + mx + tw, sy0 + my + th)

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objs = {}
        for name in self.templates:
            self._match(gray, name)
            box = self.boxes[name]
            ref = self.ref_boxes[name]
            # The anchor (top centre) moves with the box.
            ax, ay = self.anchors.get(name, ((ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2))
            objs[name] = {"box": list(box), "score": round(self.scores.get(name, 1.0), 2),
                          "anchor_px": [round(ax + box[0] - ref[0]), round(ay + box[1] - ref[1])]}
        taped = find_knife_tape(frame)
        grip = taped[2] if taped else find_gripper(frame)
        return {"label": self.label, "knife_tape": taped[:2] if taped else None, "objects": objs, "gripper_px": [round(g) for g in grip] if grip else None}


def scene_3d(trackers, cam_states, knife_state, calibration):
    """Triangulate bagel/tub tops, place the knife, and work out the distances between them."""
    world = {}
    for name in ("bagel", "tub"):
        views = [(t.P, cam_states[c]["objects"][name]["anchor_px"]) for c, t in trackers.items()
                 if t.P and c in cam_states and cam_states[c]["objects"][name]["score"] >= MIN_SCORE]
        if len(views) >= 2:
            world[name + "_top"] = [round(float(v), 1) for v in triangulate(views)]
    points = (knife_state or {}).get("points", {})
    if "knife_tip" in points and "knife_base" in points:
        world["knife_base"] = points["knife_base"]
        world["knife_tip"] = points["knife_tip"]
        world["knife_length"] = calibration.get("knife_length")
    rel = {}
    tip = np.array(world["knife_tip"]) if "knife_tip" in world else None
    for a, b in (("knife_tip", "tub_top"), ("knife_tip", "bagel_top"), ("tub_top", "bagel_top")):
        if a in world and b in world:
            d = np.array(world[b]) - np.array(world[a])
            rel[f"{a.split('_')[0]}->{b.split('_')[0]}"] = {
                "dx": round(float(d[0]), 1), "dy": round(float(d[1]), 1), "dz": round(float(d[2]), 1),
                "dist": round(float(np.linalg.norm(d)), 1)}
    if tip is not None and "bagel_top" in world:
        world["tip_above_bagel_top_cm"] = round(float(tip[2] - world["bagel_top"][2]), 1)
    if tip is not None and "tub_top" in world:
        world["tip_above_tub_rim_cm"] = round(float(tip[2] - world["tub_top"][2]), 1)
    return world, rel


def draw(tracker, frame, cam_state, world, rel):
    out = frame.copy()
    for name, o in cam_state["objects"].items():
        if o["score"] < MIN_SCORE:  # not on the table (or not found): don't draw a stale box
            continue
        x0, y0, x1, y1 = o["box"]
        cv2.rectangle(out, (x0, y0), (x1, y1), COLORS[name], 2)
        cv2.circle(out, tuple(o["anchor_px"]), 4, COLORS[name], -1)
        label = f"{name} {o['score']}"
        if name + "_top" in world:
            label += f"  ({', '.join(f'{v:.0f}' for v in world[name + '_top'])}) cm"
        cv2.putText(out, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[name], 1)
    if tracker.top_line and cam_state.get("bagel_curve"):
        curve = np.array(cam_state["bagel_curve"], np.int32)
        cv2.polylines(out, [curve], False, (0, 255, 255), 2)
        cv2.putText(out, "bagel top edge", (int(curve[0][0]) - 40, int(curve[:, 1].min()) - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if tracker.P and "tub_top" in world:
        # Where the tub must be for a straight-down dip: a circle on the table at tub-rim height.
        z = world["tub_top"][2]
        arc = [project(tracker.P, [DIP_REACH * np.cos(a), DIP_REACH * np.sin(a), z]) for a in np.linspace(-1.6, 1.6, 60)]
        arc = [p for p in arc if -2000 < p[0] < 2000 and -2000 < p[1] < 2000]
        cv2.polylines(out, [np.array(arc, np.int32)], False, (0, 255, 0), 1)
        if arc:
            mid = arc[len(arc) // 2]
            cv2.putText(out, "dip reach", (mid[0] + 4, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if tracker.P and tracker.role == "top" and "bagel_top" in world:
        z = world["bagel_top"][2]
        for radius in SPREAD_ZONE:
            arc = [project(tracker.P, [radius * np.cos(a), radius * np.sin(a), z]) for a in np.linspace(-1.2, 1.2, 50)]
            cv2.polylines(out, [np.array(arc, np.int32)], False, (255, 0, 255), 2)
        mid = project(tracker.P, [sum(SPREAD_ZONE) / 2, 0, z])
        cv2.putText(out, "spread zone: put the bagel here", (mid[0] - 90, mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    if tracker.P and "knife_tip" in world:
        base, tip = project(tracker.P, world["knife_base"]), project(tracker.P, world["knife_tip"])
        snapped = False
        if cam_state.get("knife_tape"):  # green tape on the blade: the most direct fix there is
            base, tip = cam_state["knife_tape"]
            snapped = True
        elif tracker.snap:
            base, tip, snapped = snap_knife(frame, base, tip, tracker.snap_width)
            if not snapped:  # model too far off in this view: find the blade beside the gripper
                anchor = cam_state.get("gripper_px")
                found = find_blade(frame, anchor) if anchor else find_blade(frame, base, near=260)
                if found:
                    base, tip, snapped = found[0], found[1], True
        cam_state["knife_px"] = {"base": base, "tip": tip, "on_blade": snapped}
        cv2.line(out, tuple(base), tuple(tip), COLORS["knife"], 3)
        cv2.circle(out, tuple(tip), 6, COLORS["knife"], 2)
        txt = f"knife {world['knife_length']} cm, tip ({', '.join(f'{v:.0f}' for v in world['knife_tip'])})"
        cv2.putText(out, txt, (tip[0] + 8, tip[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["knife"], 1)
        for name in ("tub", "bagel"):
            if name in cam_state["objects"]:
                cv2.line(out, tuple(tip), tuple(cam_state["objects"][name]["anchor_px"]), (200, 200, 200), 1)
    elif cam_state["gripper_px"]:
        cv2.circle(out, tuple(cam_state["gripper_px"]), 6, COLORS["knife"], 2)
        cv2.putText(out, "gripper", (cam_state["gripper_px"][0] + 8, cam_state["gripper_px"][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["knife"], 1)
    y = 18
    for key, r in rel.items():
        cv2.putText(out, f"{key}: dx {r['dx']} dy {r['dy']} dz {r['dz']}  dist {r['dist']} cm", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 16
    if tracker.role == "top" and "coverage_pct" in world:
        cv2.putText(out, f"bagel covered: {world['coverage_pct']:.0f}%", (8, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    if tracker.role == "side" and "touching" in world:
        txt = "knife TOUCHING bagel" if world["touching"] else f"knife {world.get('tip_gap_px', '?')} px above edge"
        cv2.putText(out, txt, (8, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0) if world["touching"] else (0, 200, 255), 2)
    for key in ("tip_above_bagel_top_cm", "tip_above_tub_rim_cm"):
        if key in world:
            cv2.putText(out, f"{key.replace('_cm', '').replace('_', ' ')}: {world[key]} cm", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y += 16
    cv2.putText(out, tracker.label, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def main():
    frame_dir = sys.argv[1]
    once = "--once" in sys.argv
    refs = read_json(REFS)
    calibration = read_json(CALIBRATION) or {}
    trackers = {cam: CameraTracker(cam, ref, calibration) for cam, ref in refs["cameras"].items()}
    coverage, last_save = None, 0
    try:
        while True:
            frames, cam_states = {}, {}
            for cam, t in trackers.items():
                frame = cv2.imread(os.path.join(frame_dir, f"live_{cam}.jpg"))
                if frame is not None:
                    frames[cam] = frame
                    cam_states[cam] = t.update(frame)
            world, rel = scene_3d(trackers, cam_states, read_json(KNIFE_STATE), calibration)
            side = next((c for c, t in trackers.items() if t.role == "side" and c in frames), None)
            top = next((c for c, t in trackers.items() if t.role == "top" and c in frames), None)
            if side:
                found = cam_states[side]["objects"]["bagel"]["score"] >= MIN_SCORE
                curve = bs.side_top_curve(frames[side], cam_states[side]["objects"]["bagel"]["box"]) if found else None
                cam_states[side]["bagel_curve"] = curve
                if curve and "bagel_top" in world and trackers[side].P:
                    world["bagel_profile"] = [[round(o, 2), round(z, 2)] for o, z in
                                              bs.surface_profile(trackers[side].P, curve, world["bagel_top"])]
            panel_side = cv2.resize(draw(trackers[side], frames[side], cam_states[side], world, rel), (640, 480)) if side else None
            # Touching = the real blade's tip (snapped in the side view) reaches the bagel's top edge.
            kp = cam_states.get(side, {}).get("knife_px") if side else None
            curve = cam_states.get(side, {}).get("bagel_curve") if side else None
            world["touching"] = False
            if kp and kp["on_blade"] and curve:
                us, vs = np.array([c[0] for c in curve]), np.array([c[1] for c in curve])
                u, v = kp["tip"]
                if us.min() <= u <= us.max():
                    edge_v = float(np.interp(u, us, vs))
                    world["tip_gap_px"] = int(round(edge_v - v))
                    world["touching"] = v >= edge_v - TOUCH_PX
            if top and "bagel_top" in world and trackers[top].P:
                centre = world["bagel_top"]
                if coverage is None or np.linalg.norm(coverage.origin + bs.HALF_SIZE - np.array(centre[:2])) > 3:
                    coverage = bs.Coverage(centre)  # first frame, or the bagel was moved: start over
                coverage.update_bagel(trackers[top].P, trackers[top].bagel_mask(), centre[2])
                if "knife_tip" in world:
                    coverage.paint(world["knife_tip"], world["touching"])
                world["coverage_pct"] = round(coverage.percent(), 1)
                if time.time() - last_save > 1:
                    coverage.save()
                    last_save = time.time()
            panels = []
            if top:
                p = draw(trackers[top], frames[top], cam_states[top], world, rel)
                if coverage is not None and "bagel_top" in world:  # the bagel may have been taken away
                    p = coverage.draw(p, trackers[top].P, world["bagel_top"][2])
                panels.append(cv2.resize(p, (640, 480)))
            if panel_side is not None:
                panels.append(panel_side)
            state = {"time": time.time(), "world_cm": world, "relative_cm": rel, "cameras": cam_states}
            with open(STATE + ".tmp", "w") as f:
                json.dump(state, f)
            os.replace(STATE + ".tmp", STATE)
            view = np.hstack(panels) if panels else np.zeros((480, 640, 3), np.uint8)
            if once:
                print(json.dumps({"world_cm": world, "relative_cm": rel}, indent=1))
                cv2.imwrite(os.path.join(frame_dir, "tracked.jpg"), view)
                return
            cv2.imshow("Scene tracker (q to quit)", view)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

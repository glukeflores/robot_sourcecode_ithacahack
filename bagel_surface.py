# The bagel's real top surface and how much of it the knife has covered.
#
# - side_top_curve(): the bagel's curved top edge in the side camera, found column by column as the
#   first pixel clearly brighter than the background just above it, then smoothed by a robust fit.
# - surface_profile(): that edge turned into heights (cm) by intersecting each pixel's viewing ray
#   with the vertical plane through the bagel centre, facing the side camera. A round bagel's
#   silhouette at sideways offset r is its height at distance r from the centre, so this gives
#   height(r) anywhere on the bagel.
# - Coverage: a 2.5 mm grid over the bagel top (which cells are bagel comes from the top camera).
#   Whenever the knife tip is within CONTACT_TOL of the surface, the cells along its path within
#   BRUSH_RADIUS are marked covered. Saved to coverage.npz so spreading can aim at what's left.
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COVERAGE = os.path.join(HERE, "coverage.npz")
CELL = 0.25          # cm
HALF_SIZE = 9.0      # cm: grid spans the bagel centre +/- this
BRUSH_RADIUS = 1.0   # cm: half the width of the strip one pass of the blade covers
CONTACT_TOL = 0.5    # cm: tip this close to (or below) the surface counts as touching


# ---------------------------------------------------------------- side view: the curved top edge

def side_top_curve(frame, box, margin=25):
    """Pixels [(u, v), ...] along the bagel's top edge inside the tracked box, or None."""
    x0, y0, x1, y1 = box
    h, w = frame.shape[:2]
    top = max(0, y0 - margin)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0].astype(float)
    us, vs = [], []
    for u in range(max(0, x0 + 4), min(w, x1 - 4), 2):
        col = lab[top:min(h, y1 + 5), u]
        bg = np.median(col[:8])  # background just above the bagel
        bright = col > bg + 18
        # first run of 3 bright pixels
        run = np.convolve(bright.astype(int), np.ones(3, int), "valid") == 3
        idx = np.argmax(run) if run.any() else None
        if idx is not None and idx > 3:
            us.append(u)
            vs.append(top + idx)
    if len(us) < 15:
        return None
    us, vs = np.array(us, float), np.array(vs, float)
    # Robust smooth: fit, drop points far off (knife crossing, shadows), refit.
    keep = np.ones(len(us), bool)
    for _ in range(3):
        coef = np.polyfit(us[keep], vs[keep], 4)
        keep = np.abs(np.polyval(coef, us) - vs) < 4
        if keep.sum() < 10:
            return None
    xs = np.arange(us[keep].min(), us[keep].max() + 1, 3)
    return [(int(u), int(round(np.polyval(coef, u)))) for u in xs]


def camera_centre(P):
    _, _, vt = np.linalg.svd(np.array(P))
    c = vt[-1]
    return c[:3] / c[3]


def surface_profile(P_side, curve, centre):
    """Heights along the bagel: [(signed sideways offset cm, z cm), ...] from the side-view edge."""
    P = np.array(P_side)
    C = camera_centre(P)
    look = np.array(centre) - C
    look[2] = 0
    look /= np.linalg.norm(look)
    side = np.array([-look[1], look[0], 0.0])
    Minv = np.linalg.inv(P[:, :3])
    out = []
    for u, v in curve:
        ray = Minv @ np.array([u, v, 1.0])
        denom = ray @ look
        if abs(denom) < 1e-9:
            continue
        t = ((np.array(centre) - C) @ look) / denom
        p = C + t * ray
        out.append((float((p - np.array(centre)) @ side), float(p[2])))
    return out


def height_at(profile, centre, xy, fallback_z):
    """Surface height (cm) at table position xy, assuming the bagel is round about its centre."""
    if not profile:
        return fallback_z
    r = float(np.hypot(xy[0] - centre[0], xy[1] - centre[1]))
    offs = np.array([p[0] for p in profile])
    zs = np.array([p[1] for p in profile])
    near = np.abs(np.abs(offs) - r) < 0.6  # both sides of the silhouette at this radius
    if not near.any():
        return None  # off the bagel
    return float(zs[near].max())


# ---------------------------------------------------------------- coverage

class Coverage:
    """Which parts of the bagel top the knife has been over, on a world grid around the bagel."""

    def __init__(self, centre):
        n = int(2 * HALF_SIZE / CELL)
        self.origin = np.array(centre[:2]) - HALF_SIZE
        self.covered = np.zeros((n, n), bool)
        self.bagel = np.zeros((n, n), bool)
        if os.path.exists(COVERAGE):
            d = np.load(COVERAGE)
            if d["covered"].shape == self.covered.shape and np.linalg.norm(d["origin"] - self.origin) < 3:
                self.origin, self.covered = d["origin"], d["covered"]
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self.xy = np.stack([self.origin[0] + (ii + 0.5) * CELL, self.origin[1] + (jj + 0.5) * CELL], -1)
        self.last_tip = None

    def update_bagel(self, P_top, mask, z):
        """Mark which cells are bagel: project each cell (at bagel-top height) into the top-view mask."""
        pts = np.concatenate([self.xy.reshape(-1, 2), np.full((self.xy[..., 0].size, 1), z), np.ones((self.xy[..., 0].size, 1))], 1)
        p = (np.array(P_top) @ pts.T).T
        u = np.round(p[:, 0] / p[:, 2]).astype(int)
        v = np.round(p[:, 1] / p[:, 2]).astype(int)
        h, w = mask.shape
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        inside = np.zeros(len(u), bool)
        inside[ok] = mask[v[ok], u[ok]] > 0
        self.bagel = inside.reshape(self.bagel.shape)

    def paint(self, tip, touching):
        """Mark cells under the knife's path since the last update as covered (if it's touching)."""
        tip = np.array(tip[:2])
        if touching:
            a = self.last_tip if self.last_tip is not None else tip
            ab = tip - a
            L2 = float(ab @ ab)
            rel = self.xy - a
            t = np.clip((rel @ ab) / L2, 0, 1) if L2 > 1e-9 else np.zeros(self.xy.shape[:2])
            d = np.linalg.norm(rel - t[..., None] * ab, axis=-1)
            self.covered |= d <= BRUSH_RADIUS
            self.last_tip = tip
        else:
            self.last_tip = None

    def percent(self):
        total = self.bagel.sum()
        return 100.0 * (self.covered & self.bagel).sum() / total if total else 0.0

    def save(self):
        np.savez(COVERAGE + ".tmp.npz", origin=self.origin, covered=self.covered, bagel=self.bagel, cell=CELL)
        os.replace(COVERAGE + ".tmp.npz", COVERAGE)

    def draw(self, out, P_top, z):
        """Green over covered bagel, faint outline of the rest, in the top view."""
        overlay = out.copy()
        for cells, colour in ((self.bagel & ~self.covered, (0, 140, 255)), (self.covered & self.bagel, (0, 220, 0))):
            idx = np.argwhere(cells)
            if not len(idx):
                continue
            xy = self.xy[idx[:, 0], idx[:, 1]]
            pts = np.concatenate([xy, np.full((len(xy), 1), z), np.ones((len(xy), 1))], 1)
            p = (np.array(P_top) @ pts.T).T
            for u, v in (p[:, :2] / p[:, 2:]).astype(int):
                cv2.circle(overlay, (int(u), int(v)), 2, colour, -1)
        alpha = 0.45
        return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def bagel_mask_from_box(frame, box):
    """Bagel-only mask in a frame, from its bounding box (GrabCut, so the plate is left out)."""
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    x0, y0, x1, y1 = box
    cv2.grabCut(frame, mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

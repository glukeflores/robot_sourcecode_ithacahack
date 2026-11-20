# Spread (pretend) cream cheese: the knife, blade near-vertical, goes back and forth across the bagel
# in side-to-side rows, aiming each row at the parts the coverage map still shows as uncovered.
#
#   python3 spread.py plan          # what it would do, and how much of the bagel it can reach
#   python3 spread.py run [target%] # do it (default: until 90% covered or nothing reachable is left)
#   python3 spread.py reset         # forget coverage (start a fresh bagel)
#
# Needs arm_server.py and scene_tracker.py running. Heights come from the side camera's bagel
# profile; on the first stroke the knife is lowered slowly until the side camera sees the blade
# reach the bagel's edge, and that touch height corrects the profile for the rest.
import json
import math
import os
import sys
import time

import numpy as np

import bagel_surface as bs
import knife_tasks as k

HERE = k.HERE
MAX_TILT = 8.0      # degrees the blade may lean from vertical
ROW_SPACING = 1.5   # cm between strokes (brush is 2 cm wide, so rows overlap)
STEP = 1.0          # cm between waypoints along a stroke
PRESS = 0.2         # cm below the surface, so the blade actually meets it
HOVER = 3.0         # cm above the surface when moving between strokes
PASSES = 2          # back-and-forth passes per row


def scene():
    s = json.load(open(os.path.join(HERE, "scene_state.json")))
    return s["world_cm"]


def surface(world, xy, offset):
    h = bs.height_at(world.get("bagel_profile"), world["bagel_top"], xy, world["bagel_top"][2])
    return None if h is None else h + offset


def solve(tip, j):
    sol, err, tilt = k.solve_knife(tip, j, direction=[0, 0, -1], dir_weight=4.0)
    return (sol, tilt) if err < 0.3 and tilt <= MAX_TILT else (None, tilt)


def rows(world):
    """Side-to-side rows across the bagel (perpendicular to the robot's reach), with uncovered cells."""
    cov = np.load(bs.COVERAGE)
    centre = np.array(world["bagel_top"][:2])
    radial = centre / np.linalg.norm(centre)
    side = np.array([-radial[1], radial[0]])
    n = cov["covered"].shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xy = np.stack([cov["origin"][0] + (ii + 0.5) * bs.CELL, cov["origin"][1] + (jj + 0.5) * bs.CELL], -1)
    todo = cov["bagel"] & ~cov["covered"]
    out = []
    for off in np.arange(-bs.HALF_SIZE, bs.HALF_SIZE, ROW_SPACING):
        band = todo & (np.abs((xy - centre) @ radial - off) < ROW_SPACING / 2)
        if band.sum() < 6:
            continue
        lat = (xy[band] - centre) @ side
        out.append({"radial_off": off, "from": lat.min() - 0.5, "to": lat.max() + 0.5, "cells": int(band.sum())})
    return out, centre, radial, side


def stroke_points(row, centre, radial, side):
    lats = np.arange(row["from"], row["to"] + 1e-6, STEP)
    return [centre + radial * row["radial_off"] + side * l for l in lats]


def plan():
    world = scene()
    j = {int(a): b for a, b in k.arm_cmd("state")["joints"].items()}
    rs, centre, radial, side = rows(world)
    total = reach = 0
    for row in rs:
        pts = stroke_points(row, centre, radial, side)
        ok = 0
        for p in pts:
            z = surface(world, p, -PRESS)
            if z is None:
                continue
            sol, tilt = solve([p[0], p[1], z], j)
            if sol:
                ok += 1
                j = sol
        total += len(pts)
        reach += ok
        print(f"row {row['radial_off']:+.1f} cm: {row['cells']} uncovered cells, {ok}/{len(pts)} stroke points reachable")
    print(f"reachable: {reach}/{total} stroke points ({100 * reach / max(total, 1):.0f}%) "
          f"with the blade within {MAX_TILT} deg of vertical; coverage now {world.get('coverage_pct', 0)}%")


def move_to(tip, j, secs=1.5):
    sol, tilt = solve(tip, j)
    if sol is None:
        raise RuntimeError(f"can't reach {np.round(tip, 1)} with the blade within {MAX_TILT} deg of vertical ({tilt:.0f} deg)")
    st = k.arm_cmd("joints", joints={str(s): v for s, v in sol.items()}, seconds=secs)
    return {int(a): b for a, b in st["joints"].items()}


def find_touch(world, j):
    """Lower over the bagel centre until the side camera sees the blade touch; return the correction."""
    c = np.array(world["bagel_top"][:2])
    z0 = surface(world, c, 0.0)
    for dz in np.arange(2.0, -2.01, -0.25):
        j = move_to([c[0], c[1], z0 + dz], j, 0.6)
        time.sleep(0.6)
        w = scene()
        if w.get("touching"):
            print(f"touch seen at {dz:+.2f} cm vs the profile estimate")
            return dz, j
    print("no touch seen down to -2 cm; using the profile as is")
    return 0.0, j


def run(target):
    j = {int(a): b for a, b in k.arm_cmd("state")["joints"].items()}
    world = scene()
    c = world["bagel_top"]
    j = move_to([c[0], c[1], surface(world, c[:2], HOVER)], j, 3.0)  # hover over the centre first
    offset, j = find_touch(world, j)
    while True:
        world = scene()
        if world.get("coverage_pct", 0) >= target:
            break
        rs, centre, radial, side = rows(world)
        rs = sorted(rs, key=lambda r: -r["cells"])
        did = False
        for row in rs:
            pts = [p for p in stroke_points(row, centre, radial, side) if surface(world, p, 0) is not None]
            path, jj = [], dict(j)
            for p in pts:
                sol, _ = solve([p[0], p[1], surface(world, p, offset - PRESS)], jj)
                if sol:
                    path.append(sol)
                    jj = sol
            if len(path) < 3:
                continue
            start = pts[0]
            j = move_to([start[0], start[1], surface(world, start, offset + HOVER)], j, 1.5)
            for n in range(PASSES):
                legs = path if n % 2 == 0 else path[::-1]
                k.arm_cmd("path", waypoints=[{str(s): v for s, v in w.items()} for w in legs], seconds_each=0.35)
            end = pts[-1] if PASSES % 2 else pts[0]
            j = move_to([end[0], end[1], surface(world, end, offset + HOVER)], legs[-1], 1.0)
            time.sleep(1.0)
            print(f"row {row['radial_off']:+.1f} cm done; coverage {scene().get('coverage_pct', 0)}%", flush=True)
            did = True
            break
        if not did:
            print("nothing reachable left to cover")
            break
    world = scene()
    c = world["bagel_top"]
    move_to([c[0], c[1], surface(world, c[:2], offset + 6)], j, 2.0)
    print(f"finished: bagel {world.get('coverage_pct', 0)}% covered")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "reset":
        if os.path.exists(bs.COVERAGE):
            os.remove(bs.COVERAGE)
        print("coverage cleared (restart scene_tracker.py to start from 0%)")
    elif cmd == "plan":
        plan()
    elif cmd == "run":
        run(float(sys.argv[2]) if len(sys.argv) > 2 else 90.0)

# Owns the knife arm's serial port: publishes where the arm (and knife) is ~10x a second to
# knife_state.json, and carries out move commands dropped into arm_cmds/ by other scripts
# (knife_tasks.arm_cmd). Only one process may talk to the servos at a time, so everything goes
# through here and the live view always has a fresh knife position.
#
#   python3 arm_server.py          # Ctrl+C stops it; the arm keeps holding its pose
import glob
import json
import os
import termios
import time

import serial

import numpy as np

import knife_tasks as k
import robot_up_down as r

CMD_DIR = os.path.join(k.HERE, "arm_cmds")
PUBLISH_EVERY = 0.1
FAST_SPEED = 700      # ticks/second cap for commanded moves (robot_up_down's MOVE_SPEED is 300)
SETTLE = 0.1          # seconds to let the arm settle after a move
SAG_JOINTS = (r.LIFT, r.ELBOW, r.WRIST)  # joints that hold the arm's weight
SAG_TOLERANCE = 6     # ticks (~0.5 degrees) of sag that's left alone
SAG_MAX = 80          # never aim more than this many ticks past a target
SAG_PASSES = 2


def knife_points(j):
    """World positions of the calibrated points on the gripper/knife, if measured."""
    cal = k.load_calibration()
    return {name: [round(float(v), 2) for v in k.hand_point(j, off)] for name, off in cal.get("tool", {}).items()}


def publish(arm, j=None):
    j = j or {sid: arm.read_position(sid) for sid in r.ALL_JOINTS}
    wrist, pitch = k.forward(j)
    state = {"time": time.time(), "joints": j, "wrist": [round(v, 2) for v in wrist],
             "pitch": round(pitch, 1), "points": knife_points(j)}
    tmp = k.KNIFE_STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, k.KNIFE_STATE)
    return state


SAG_MEMORY = {}  # (joint, direction) -> how far it sagged last time; aimed past from the start


def move(arm, target, seconds, speed=FAST_SPEED):
    """Ease to joint targets with the contact guard, publishing along the way. Joints that carry
    the arm's weight are aimed past their target by the sag they showed last time."""
    now = {sid: arm.read_position(sid) for sid in target}
    goal_target = dict(target)
    target = dict(target)
    for s in SAG_JOINTS:
        if s in target and target[s] != now[s]:
            target[s] += SAG_MEMORY.get((s, np.sign(target[s] - now[s])), 0)
    distance = max(abs(target[s] - now[s]) for s in target)
    steps = max(1, int(max(seconds, distance / speed) / 0.02))
    last = 0
    for i in range(1, steps + 1):
        f = (1 - np.cos(np.pi * i / steps)) / 2
        goals = {s: now[s] + (target[s] - now[s]) * f for s in target}
        for s, g in goals.items():
            arm.write_position(s, g)
        arm.check(goals)
        if time.time() - last > PUBLISH_EVERY:
            publish(arm)
            last = time.time()
        time.sleep(0.02)
    time.sleep(SETTLE)
    compensate_sag(arm, goal_target, target, now)
    return publish(arm)


def compensate_sag(arm, target, aimed=None, start=None):
    """Gravity holds the lifting joints a little short of their goal (the servo needs some error
    to push back). Re-aim each one past its target by what it fell short, so it settles on it,
    and remember the total offset that was needed for the next move in the same direction."""
    aimed = aimed or target
    goals = {s: aimed[s] for s in SAG_JOINTS if s in target}
    for _ in range(SAG_PASSES):
        short = {s: target[s] - arm.read_position(s) for s in goals}
        if max((abs(v) for v in short.values()), default=0) <= SAG_TOLERANCE:
            break
        for s, err in short.items():
            if abs(err) > SAG_TOLERANCE:
                goals[s] = int(np.clip(goals[s] + err, target[s] - SAG_MAX, target[s] + SAG_MAX))
                arm.write_position(s, goals[s])
        time.sleep(0.15)
    if start:
        for s in goals:
            if target[s] != start[s]:
                SAG_MEMORY[(s, np.sign(target[s] - start[s]))] = int(goals[s] - target[s])


def follow(arm, waypoints, seconds_each):
    """Glide through joint waypoints without stopping at each (linear between them), guarded."""
    pts = [{s: arm.read_position(s) for s in waypoints[0]}] + waypoints
    last = 0
    for a, b in zip(pts, pts[1:]):
        steps = max(1, int(seconds_each / 0.02))
        for i in range(1, steps + 1):
            f = i / steps
            goals = {s: a[s] + (b[s] - a[s]) * f for s in b}
            for s, g in goals.items():
                arm.write_position(s, g)
            arm.check(goals)
            if time.time() - last > PUBLISH_EVERY:
                publish(arm)
                last = time.time()
            time.sleep(0.02)
    time.sleep(0.3)
    return publish(arm)


def handle(arm, cmd):
    op = cmd["op"]
    if op == "state":
        return publish(arm)
    if op == "joints":
        return move(arm, {int(s): int(v) for s, v in cmd["joints"].items()}, cmd.get("seconds", 1.0),
                    cmd.get("speed", FAST_SPEED))
    if op == "path":
        return follow(arm, [{int(k_): int(v) for k_, v in w.items()} for w in cmd["waypoints"]], cmd.get("seconds_each", 0.4))
    if op == "goto":
        target = k.inverse(cmd["x"], cmd["y"], cmd["z"], cmd["pitch"])
        if "roll" in cmd:
            target[r.ROLL] = int(cmd["roll"])
        return move(arm, target, cmd.get("seconds", 2.5))
    if op == "release":
        # Fade the torque out over ~3 s so the arm settles instead of dropping, then switch off.
        for cap in range(r.TORQUE_CAP, -1, -35):
            for sid in r.ALL_JOINTS:
                arm.write_word(sid, r.TORQUE_LIMIT, cap)
            time.sleep(0.15)
        arm.release()
        return publish(arm)
    if op == "hold":
        arm.hold()
        return publish(arm)
    raise ValueError(f"unknown op {op}")


LINK_ERRORS = (serial.SerialException, OSError, termios.error)  # USB unplugged/bumped


def connect():
    """Open the arm, retrying until its USB port is back."""
    while True:
        try:
            arm = r.Arm(k.PORT)
            arm.hold()
            return arm
        except (*LINK_ERRORS, RuntimeError) as e:
            print(f"waiting for the arm ({e})", flush=True)
            time.sleep(1)


def serve_one(arm, last):
    """Run the next queued command, or publish the arm state. Returns the last publish time."""
    pending = sorted(glob.glob(os.path.join(CMD_DIR, "*.cmd.json")))
    if not pending:
        if time.time() - last > PUBLISH_EVERY:
            publish(arm)
            return time.time()
        time.sleep(0.02)
        return last
    path = pending[0]
    with open(path) as f:
        cmd = json.load(f)
    os.remove(path)
    result = {"ok": False, "error": "interrupted"}
    try:
        result = {"ok": True, "result": handle(arm, cmd)}
    except r.Contact as e:
        arm.freeze()
        result = {"ok": False, "error": f"contact: {e}"}
    except LINK_ERRORS:
        result = {"ok": False, "error": "lost the USB link to the arm mid-command; reconnecting"}
        raise
    except Exception as e:  # noqa: BLE001 - report every failure back to the caller
        result = {"ok": False, "error": str(e)}
    finally:
        out = path.replace(".cmd.json", ".result.json")
        with open(out + ".tmp", "w") as f:
            json.dump(result, f)
        os.replace(out + ".tmp", out)
    return last


def main():
    os.makedirs(CMD_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(CMD_DIR, "*")):
        os.remove(stale)
    arm = connect()
    print("arm server running", flush=True)
    last = 0
    try:
        while True:
            try:
                last = serve_one(arm, last)
            except LINK_ERRORS as e:
                print(f"lost the arm's USB link ({e}); reconnecting", flush=True)
                try:
                    arm.close()
                except LINK_ERRORS:
                    pass
                time.sleep(1)
                arm = connect()
                print("reconnected", flush=True)
    except KeyboardInterrupt:
        print("arm server stopped; the arm is still holding its pose")
    finally:
        arm.close()


if __name__ == "__main__":
    main()

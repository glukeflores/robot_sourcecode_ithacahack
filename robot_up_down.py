# Moves every connected SO-ARM101 (Feetech STS3215 servos) up to full extension and back down,
# rotating the hand and opening the gripper on the way up, closing it on the way down.
#
# Step 1, find the straight-up pose automatically (all arms at once, ~3 minutes):
#   python3 robot_up_down.py calibrate
#   Shoulder, elbow and wrist are each swept slowly up and back down while reading servo load.
#   Averaging the two directions cancels friction and leaves pure gravity, a sine wave in the
#   joint angle whose zero is where everything above that joint points straight up. Fitting the
#   wave finds that angle; the shoulder and elbow are refined until they agree, then the arm
#   returns to where it started.
# Step 2, run it. Each arm rises from wherever it is resting to the calibrated pose and back:
#   python3 robot_up_down.py
#   python3 robot_up_down.py --period 8 --hold 2 --cycles 3 --roll 180 --grip 0.6
#
# All six joints are held under power the whole time. If any joint meets resistance (the table,
# an object, a person) the arm stops: if it was rising it goes back the way it came, if it was
# lowering it settles where it is. Ctrl+C eases every arm back to where it started and then
# releases torque.
import argparse
import glob
import json
import math
import os
import threading
import time

import serial

PORT_PATTERN = "/dev/tty.usbmodem*"  # every arm's USB servo board shows up as one of these
BAUD = 1_000_000
PAN, LIFT, ELBOW, WRIST, ROLL, GRIPPER = 1, 2, 3, 4, 5, 6
ALL_JOINTS = [PAN, LIFT, ELBOW, WRIST, ROLL, GRIPPER]
LIFT_JOINTS = [LIFT, ELBOW, WRIST]
GUARDED = [PAN, LIFT, ELBOW, WRIST, ROLL]  # the gripper has its own low torque limit instead
JOINT_NAMES = {PAN: "base", LIFT: "shoulder", ELBOW: "elbow", WRIST: "wrist", ROLL: "wrist roll", GRIPPER: "gripper"}
POSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_up_pose.json")

# Feetech STS3215 control table
MIN_LIMIT = 9
MAX_LIMIT = 11
TORQUE_ENABLE = 40
ACCELERATION = 41
GOAL_POSITION = 42
TORQUE_LIMIT = 48
PRESENT_POSITION = 56
PRESENT_LOAD = 60
WRITE, READ = 0x03, 0x02

# Safety
TORQUE_CAP = 700      # servo-enforced max torque for arm joints (per mille), so contact can't crush
LIFT_TORQUE_CAP = 850  # shoulder and elbow hold the arm up against gravity: give them more to lift fast
ACCEL = 60            # servo acceleration setting (was 20): quicker starts, less lag on fast moves
GRIPPER_TORQUE = 350  # gentler cap for the gripper so it can't squeeze hard
CONTACT_LOAD = 480    # a joint pushing harder than this has hit something (normal moves peak ~340)
ROLL_CONTACT_LAG = 600  # the wrist roll is light and fast-turning: allow more lag before calling it a hit
ROLL_CONTACT_LOAD = 650  # ...and more load (spinning the knife up quickly takes a burst; the 70% torque cap still applies)
MOVE_SPEED = 300      # average ticks/second for repositioning moves (as fast as the up/down motion)
CONTACT_LAG = 220     # ...or one this far behind its goal (~19 degrees; a fast shoulder lags ~145)

# Calibration tuning
STEP = 10             # ticks per sweep step (~0.9 degrees)
SETTLE = 0.05         # seconds per step
SHOULDER_SWEEP = 0.65  # fraction of its range to sweep from rest toward mid-range
# The elbow is swept from fully extended to ~80 degrees of bend, never further: with the upper arm
# upright, bending more points the hand back down at the table.
ELBOW_SWEEP = 0.42
WRIST_SWEEP = 0.3     # +/- this fraction of its range around mid-range
REFINE = 0.2          # later shoulder passes sweep +/- this fraction around the last answer
MIN_AMPLITUDE = 6     # weakest gravity wave (load per mille) worth trusting
MIN_FIT = 0.3         # how well the sine wave must explain the readings (R^2)
SKIP_AFTER_TURN = 3   # readings to drop at the start of each sweep direction
MAX_ROUNDS = 3        # shoulder/elbow refinement rounds
SETTLED = 25          # ...stop once the shoulder moves less than this many ticks (~2 degrees)
WRIST_SWEEPS = 3
TICK = 2 * math.pi / 4096  # radians per servo tick


class Contact(Exception):
    """A joint met resistance it shouldn't have."""


class CalibrationError(Exception):
    pass


STOP = threading.Event()  # set on Ctrl+C so calibration threads bring their arms home


class Arm:
    def __init__(self, port):
        self.port = port
        self.name = "arm " + port[-5:]
        self.ser = serial.Serial(port, BAUD, timeout=0.02)
        self.strikes = 0

    def _send(self, sid, instr, params):
        body = [sid, len(params) + 2, instr, *params]
        self.ser.reset_input_buffer()
        self.ser.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))
        head = self.ser.read(4)
        if len(head) == 4 and head[:2] == b"\xff\xff":
            return head + self.ser.read(head[3])
        return None

    def read_bytes(self, sid, addr, n):
        reply = self._send(sid, READ, [addr, n])
        if not reply or len(reply) < 6 + n:
            raise RuntimeError(f"servo {sid} on {self.port} did not answer")
        return reply[5:5 + n]

    def read_word(self, sid, addr):
        b = self.read_bytes(sid, addr, 2)
        return b[0] | (b[1] << 8)

    def read_position(self, sid):
        return self.read_word(sid, PRESENT_POSITION)

    @staticmethod
    def _signed_load(raw):
        """Positive = motor pushing toward higher positions."""
        return (raw & 1023) if raw & 1024 else -(raw & 1023)

    def read_state(self, sid):
        """Position and signed load in one round trip."""
        b = self.read_bytes(sid, PRESENT_POSITION, 6)
        return b[0] | (b[1] << 8), self._signed_load(b[4] | (b[5] << 8))

    def read_effort(self, sid, samples=3):
        return sum(self._signed_load(self.read_word(sid, PRESENT_LOAD)) for _ in range(samples)) / samples

    def limits(self, sid):
        return self.read_word(sid, MIN_LIMIT), self.read_word(sid, MAX_LIMIT)

    def write_byte(self, sid, addr, value):
        self._send(sid, WRITE, [addr, value])

    def write_word(self, sid, addr, value):
        self._send(sid, WRITE, [addr, value & 0xFF, value >> 8])

    def write_position(self, sid, pos):
        self.write_word(sid, GOAL_POSITION, max(0, min(4095, int(round(pos)))))

    def hold(self, joints=ALL_JOINTS):
        """Enable torque at the current pose (so nothing jumps), with the torque caps in place."""
        pose = {sid: self.read_position(sid) for sid in joints}
        for sid in joints:
            self.write_position(sid, pose[sid])
            self.write_byte(sid, ACCELERATION, ACCEL)
            cap = GRIPPER_TORQUE if sid == GRIPPER else LIFT_TORQUE_CAP if sid in (LIFT, ELBOW) else TORQUE_CAP
            self.write_word(sid, TORQUE_LIMIT, cap)
            self.write_byte(sid, TORQUE_ENABLE, 1)
        self.strikes = 0
        return pose

    def check(self, goals):
        """Raise Contact if a guarded joint is straining or stuck. Two bad readings in a row needed."""
        for sid, goal in goals.items():
            if sid not in GUARDED:
                continue
            pos, load = self.read_state(sid)
            lag = round(abs(goal - pos))
            if abs(load) > (ROLL_CONTACT_LOAD if sid == ROLL else CONTACT_LOAD) or \
                    lag > (ROLL_CONTACT_LAG if sid == ROLL else CONTACT_LAG):
                self.strikes += 1
                if self.strikes >= 2:
                    raise Contact(f"{self.name}: {JOINT_NAMES[sid]} hit something "
                                  f"(load {load}/1000, {lag} ticks short of its goal)")
                return
        self.strikes = 0

    def freeze(self, joints=ALL_JOINTS):
        """Stop pushing: make wherever each joint is right now its goal."""
        for sid in joints:
            try:
                self.write_position(sid, self.read_position(sid))
            except RuntimeError:
                pass

    def move_to(self, pose, seconds):
        move_all([self], [pose], seconds)

    def release(self, joints=ALL_JOINTS):
        for sid in joints:
            self.write_byte(sid, TORQUE_ENABLE, 0)

    def close(self):
        self.ser.close()


def find_arms():
    """Open every USB servo board whose six servos all answer; report the ones that don't."""
    arms = []
    for port in sorted(glob.glob(PORT_PATTERN)):
        arm = Arm(port)
        try:
            for sid in ALL_JOINTS:
                arm.read_position(sid)
            arms.append(arm)
        except RuntimeError as e:
            print(f"Skipping {port}: {e} (is the servo power supply plugged in?)")
            arm.close()
    print(f"Found {len(arms)} arm(s): {', '.join(a.port for a in arms) or 'none'}")
    return arms


def move_all(arms, poses, seconds):
    """Ease several arms to their poses together (taking at least `seconds`, longer for big moves),
    stopping if any of them hits something."""
    nows = [{sid: arm.read_position(sid) for sid in pose} for arm, pose in zip(arms, poses)]
    distance = max(abs(pose[sid] - now[sid]) for pose, now in zip(poses, nows) for sid in pose)
    steps = max(1, int(max(seconds, distance / MOVE_SPEED) / 0.02))
    for i in range(1, steps + 1):
        f = (1 - math.cos(math.pi * i / steps)) / 2
        for arm, pose, now in zip(arms, poses, nows):
            goals = {sid: now[sid] + (pose[sid] - now[sid]) * f for sid in pose}
            for sid, goal in goals.items():
                arm.write_position(sid, goal)
            arm.check(goals)
        time.sleep(0.02)


def go_home(arms, starts, log):
    """Back to the starting poses together, then limp. If something is in the way, settle there instead."""
    try:
        move_all(arms, starts, 4)
        time.sleep(0.5)
    except Contact as e:
        log(f"{e}; letting the arms settle where they are instead")
        for arm in arms:
            arm.freeze()
        time.sleep(0.5)
    for arm in arms:
        arm.release()


# ---------------------------------------------------------------- calibration

def sweep(arm, sid, a, b):
    """Move joint `sid` from a to b and back, returning (positions, gravity effort at each)."""
    arm.move_to({sid: a}, 1.5)
    d = 1 if b > a else -1
    grid = list(range(a, b + d, d * STEP))
    readings = {}
    for leg in (grid, grid[::-1]):
        for n, goal in enumerate(leg):
            if STOP.is_set():
                raise CalibrationError("aborted")
            arm.write_position(sid, goal)
            time.sleep(SETTLE)
            arm.check({sid: goal})
            effort = arm.read_effort(sid, samples=2)
            if n >= SKIP_AFTER_TURN:  # the joint is still getting going; load reads low
                readings.setdefault(goal, []).append(effort)
    # Friction pushes the opposite way on the way back, so the average is gravity alone.
    grid = [g for g in grid if len(readings.get(g, ())) == 2]
    gravity = [sum(readings[g]) / 2 for g in grid]
    # Smooth out servo noise.
    smooth = [sum(gravity[max(0, i - 3):i + 4]) / len(gravity[max(0, i - 3):i + 4]) for i in range(len(grid))]
    return grid, smooth


def fit_gravity(grid, g):
    """Least-squares fit g = p*cos(t) + q*sin(t), with t the joint angle in radians.

    No constant term: friction is already cancelled, and over a short sweep a free offset
    trades off against the wave and lets the zero wander.
    """
    cs = [math.cos(x * TICK) for x in grid]
    sn = [math.sin(x * TICK) for x in grid]
    a11, a12, a22 = sum(c * c for c in cs), sum(c * s for c, s in zip(cs, sn)), sum(s * s for s in sn)
    b1, b2 = sum(c * y for c, y in zip(cs, g)), sum(s * y for s, y in zip(sn, g))
    det = a11 * a22 - a12 * a12
    p, q = (a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det
    fit = [p * c + q * s for c, s in zip(cs, sn)]
    mean = sum(g) / len(g)
    total = sum((y - mean) ** 2 for y in g) or 1e-9
    r2 = 1 - sum((y - f) ** 2 for y, f in zip(g, fit)) / total
    return p, q, r2


def find_balance(arm, sid, a, b, limits, log, sweeps=1):
    """Angle where everything above this joint points straight up.

    Gravity's pull on a joint is a sine wave in its angle: biggest when the link is horizontal,
    zero when it's straight up or straight down. Fitting that wave to the whole sweep finds the
    straight-up angle even when it lies past the end of the sweep or the signal is noisy.
    Below the top the motor pushes toward higher positions, above it toward lower ones, so the
    top is the zero where the wave goes from + to -: angle = phase + 90 degrees.
    """
    grid, g = [], []
    for _ in range(sweeps):  # light joints get several sweeps so the noise averages out
        x, y = sweep(arm, sid, a, b)
        grid += x
        g += y
    p, q, r2 = fit_gravity(grid, g)
    amplitude = math.hypot(p, q)
    if amplitude < MIN_AMPLITUDE or r2 < MIN_FIT:
        raise CalibrationError(
            f"{JOINT_NAMES[sid]}: gravity signal too weak to trust in {a}..{b} "
            f"(amplitude {amplitude:.0f}, fit {r2:.2f})")
    up = (math.atan2(q, p) + math.pi / 2) / TICK
    centre = (a + b) / 2
    up += round((centre - up) / 4096) * 4096  # pick the turn nearest the sweep
    lo, hi = limits
    best = max(lo, min(hi, round(up)))
    note = "" if best == round(up) else f" (true vertical {round(up)} is past its limit)"
    log(f"{JOINT_NAMES[sid]}: straight up at {best}{note}  [amplitude {amplitude:.0f}, fit {r2:.2f}, swept {a}..{b}]")
    arm.move_to({sid: best}, 1.5)
    return best


def calibrate_arm(arm, results):
    def log(msg):
        print(f"[{arm.name}] {msg}", flush=True)

    start = None
    try:
        limits = {sid: arm.limits(sid) for sid in LIFT_JOINTS}
        start = arm.hold()  # every joint powered, so the hand and base can't flop around
        log(f"start pose {start}")

        def clamp(sid, x):
            lo, hi = limits[sid]
            return max(lo, min(hi, round(x)))

        def around(sid, centre, frac):
            lo, hi = limits[sid]
            return clamp(sid, centre - frac * (hi - lo)), clamp(sid, centre + frac * (hi - lo))

        def elbow_window():
            # Elbow positions grow as it bends (its calibrated minimum is fully extended).
            lo, hi = limits[ELBOW]
            return lo, clamp(ELBOW, lo + ELBOW_SWEEP * (hi - lo))

        up = {}
        # Shoulder first: sweep from where it rests toward mid-range, which lifts the arm.
        lo, hi = limits[LIFT]
        here = clamp(LIFT, start[LIFT])
        d = 1 if (lo + hi) / 2 >= here else -1
        up[LIFT] = find_balance(arm, LIFT, here, clamp(LIFT, here + d * SHOULDER_SWEEP * (hi - lo)), limits[LIFT], log)
        up[ELBOW] = find_balance(arm, ELBOW, *elbow_window(), limits[ELBOW], log)
        # Shoulder and elbow depend on each other, so alternate until the shoulder settles.
        for _ in range(MAX_ROUNDS):
            before = up[LIFT]
            up[LIFT] = find_balance(arm, LIFT, *around(LIFT, up[LIFT], REFINE), limits[LIFT], log)
            up[ELBOW] = find_balance(arm, ELBOW, *elbow_window(), limits[ELBOW], log)
            if abs(up[LIFT] - before) <= SETTLED:
                break
        # Wrist last, with everything below it upright; its pull is weak, so average several sweeps.
        mid = sum(limits[WRIST]) / 2
        up[WRIST] = find_balance(arm, WRIST, *around(WRIST, mid, WRIST_SWEEP), limits[WRIST], log, sweeps=WRIST_SWEEPS)
        log(f"fully extended pose {up}")
        results[arm.port] = up
        time.sleep(2)
    except Contact as e:
        log(f"STOPPED: {e}")
        arm.freeze()
        time.sleep(0.5)
    except CalibrationError as e:
        log(f"FAILED: {e}")
    except Exception as e:
        log(f"ERROR: {e}")
    finally:
        if start:
            log("returning to start")
            go_home([arm], [start], log)


def calibrate():
    arms = find_arms()
    if not arms:
        return
    results = {}
    print("Calibrating. Keep clear; Ctrl+C to abort.")
    threads = [threading.Thread(target=calibrate_arm, args=(arm, results), daemon=True) for arm in arms]
    try:
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nAborting: bringing the arms back to start...")
        STOP.set()
        for t in threads:
            t.join()
        return
    finally:
        for arm in arms:
            arm.close()
    save_poses(results)


def save_poses(new_poses):
    poses = {}
    if os.path.exists(POSE_FILE):
        with open(POSE_FILE) as f:
            poses = json.load(f)
    poses.update({port: {str(sid): pos for sid, pos in pose.items()} for port, pose in new_poses.items()})
    with open(POSE_FILE, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"Saved up pose for {len(new_poses)} arm(s) to {os.path.basename(POSE_FILE)}")


def record():
    arms = find_arms()
    if not arms:
        return
    try:
        input("The arms are about to go limp. Hold them so they don't drop, then press Enter...")
        for arm in arms:
            arm.release()
        input("Now move every arm by hand into the highest, fully extended pose. Press Enter when they're there...")
        save_poses({arm.port: {sid: arm.read_position(sid) for sid in LIFT_JOINTS} for arm in arms})
        print("Lower the arms back to rest by hand, then run:  python3 robot_up_down.py")
    finally:
        for arm in arms:
            arm.close()


# ---------------------------------------------------------------- running

def hand_targets(arm, start, roll_degrees, grip):
    """Where the wrist roll and gripper go at the top: a twist, and an open hand."""
    targets = {}
    lo, hi = arm.limits(ROLL)
    turn = round(roll_degrees / 360 * 4096)
    # Twist whichever way stays inside the joint's range.
    targets[ROLL] = start[ROLL] + turn if start[ROLL] + turn <= hi else start[ROLL] - turn
    targets[ROLL] = max(lo, min(hi, targets[ROLL]))
    # The gripper rests closed, at the end of its range nearest where it starts; open toward the other end.
    lo, hi = arm.limits(GRIPPER)
    closed_at_min = start[GRIPPER] - lo <= hi - start[GRIPPER]
    targets[GRIPPER] = round(lo + grip * (hi - lo)) if closed_at_min else round(hi - grip * (hi - lo))
    return targets


def lift_fraction(t, period, hold):
    """0 at rest -> 1 at the top -> 0 again; eases in and out, pausing `hold` seconds at the top."""
    rise = period / 2
    t %= period + hold
    if t < rise:
        return (1 - math.cos(math.pi * t / rise)) / 2
    if t < rise + hold:
        return 1.0
    return (1 + math.cos(math.pi * (t - rise - hold) / rise)) / 2


def run(args):
    if not os.path.exists(POSE_FILE):
        print("No up pose yet. Run:  python3 robot_up_down.py calibrate")
        return
    with open(POSE_FILE) as f:
        saved = json.load(f)
    arms = []
    for arm in find_arms():
        if arm.port in saved:
            arms.append(arm)
        else:
            print(f"{arm.name} has no up pose yet; run `python3 robot_up_down.py calibrate`. Skipping it.")
            arm.close()
    if not arms:
        return

    starts, targets = [], []
    try:
        for arm in arms:
            start = arm.hold()
            # Base stays put; the arm straightens up; the hand twists and opens.
            target = {PAN: start[PAN], **{int(sid): pos for sid, pos in saved[arm.port].items()}}
            target.update(hand_targets(arm, start, args.roll, args.grip))
            starts.append(start)
            targets.append(target)
            print(f"{arm.name}: start {start}\n{' ' * len(arm.name)}  up    {target}")

        print("Moving. Press Ctrl+C to stop.")
        cycle = args.period + args.hold
        t0, last_f = time.time(), 0.0
        while True:
            t = time.time() - t0
            if args.cycles and t >= args.cycles * cycle:
                break
            f = lift_fraction(t, args.period, args.hold)
            for arm, start, target in zip(arms, starts, targets):
                goals = {sid: start[sid] + (target[sid] - start[sid]) * f for sid in ALL_JOINTS}
                for sid, goal in goals.items():
                    arm.write_position(sid, goal)
                arm.check(goals)
            last_f = f
            time.sleep(0.02)
    except Contact as e:
        rising = f > last_f
        print(f"\n{e}.")
        for arm in arms:
            arm.freeze()
        time.sleep(0.5)
        if not rising:
            # Lowering onto something: don't keep pushing into it.
            print("It was lowering, so the arms stay where they are and go limp.")
            for arm in arms:
                arm.release()
                arm.close()
            return
        print("It was rising, so the arms go back the way they came.")
    except KeyboardInterrupt:
        pass

    print("\nReturning to start and releasing torque...")
    go_home(arms, starts, print)
    for arm in arms:
        arm.close()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Move every connected arm up to full extension and back.")
    parser.add_argument("command", nargs="?", choices=("run", "calibrate", "record"), default="run")
    parser.add_argument("--period", type=float, default=8.0, help="seconds to go up and come back down")
    parser.add_argument("--hold", type=float, default=1.0, help="seconds to pause at the top")
    parser.add_argument("--cycles", type=int, default=0, help="stop after this many cycles (0 = until Ctrl+C)")
    parser.add_argument("--roll", type=float, default=90, help="degrees to twist the hand at the top")
    parser.add_argument("--grip", type=float, default=0.7, help="how far to open the gripper, 0 (closed) to 1 (fully open)")
    args = parser.parse_args()
    {"calibrate": calibrate, "record": record}.get(args.command, lambda: run(args))()


if __name__ == "__main__":
    main()

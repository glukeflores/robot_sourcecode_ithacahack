r"""Robot command center: live vision from both cameras plus manual and automatic knife control.

    python3 robot_command_center.py              # starts helpers, then moves the knife above the tub
    python3 robot_command_center.py --no-start   # don't move on start-up
    python3 robot_command_center.py --no-ai      # skip the YOLO-World detector (faster start)

On start it launches whatever isn't already running (camera_capture for both cameras, and
arm_server.py for the knife arm), waits until both cameras see the tub, then puts the knife 1 cm
above the tub (knife_center.py). From then on you drive it:

  On screen   joystick pad   drag it: up = forward, sideways = left/right (hold to keep moving)
              up/down bar    drag it up or down          roll bar   drag left or right
              step slider    0.5 - 10 cm per move tick   roll speed slider  1x - 10x
              buttons        Knife -> tub (tip 1 cm below the tub's centre), Knife -> bagel (tip to
                             the bagel's centre), Point down, STOP, and Tub -> bagel: knife to tub,
                             roll 2 s, up 3 cm, knife down to the bagel, roll 2 s
  Keyboard    W/S  forward/back      A/D  left/right      R/F  up/down
              Z/C  roll              1-5 step 0.5/1/2/5/10 cm
              T    knife to tub      B    knife to bagel  V    point blade straight down
              G    tub -> bagel sequence   H    knife flat (blade horizontal, tip stays put)
              Space stop             Q / Esc quit
  Controller  left stick  forward/back, left/right      right stick  up/down
              LB / RB     roll                          A tub   B bagel   X point down   Y stop

The camera views update ~10x a second with a trail of where the knife tip has been, and the
panel shows the knife tip's position, blade angle and joint positions live, also mid-move.

Vision: each camera shows the tracked tub and bagel (template matching, scene_refs.json) and the
green-taped knife. The top view also gets the YOLO-World detections, cream cheese coverage and
claw/blade candidates from live_bagel_robot_vision_v2.py. Those are estimates; a 2D camera can't
prove contact.

Every arm move goes through arm_server.py's contact guard and torque caps. Only one automatic
action runs at a time; Stop halts it after the move in progress (moves are at most ~2 s).
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

import knife_center as kc
import knife_tasks as k
import scene_tracker as st

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = kc.FRAME_DIR
TOP, SIDE = kc.TOP, kc.SIDE
PANEL_W, PANEL_H, BAR_H = 640, 480, 250
WINDOW = "Robot command center"
ROLL_STEP = 100          # servo ticks (~9 degrees) per roll tick at roll speed 1x (slider goes to 10x)
ROLL_SPEED = 1400        # ticks/second cap for roll moves (fast, without a torque spike)
JOG_SPEED = 900          # ticks/second cap for manual moves (big steps of up to 10 cm)
TUB_BELOW_CENTRE_CM = 1.0  # 'Knife -> tub' aims this far below the centre of the tub's box
SEQUENCE_LIFT_CM = 3.0     # 'Tub -> bagel': how far to lift out of the tub before going to the bagel
TICK_SECONDS = 0.05      # how often a held joystick/slider sends the next move
PAD_DEADZONE = 0.15
TRAIL_LEN = 40
JOG_SECONDS = 0.3       # manual move time
JOG_UP_SECONDS = 0.2    # lifting is quickest: gravity already slows the head on the way up
JOG_TRIES = 2           # corrective moves if a jog still falls short
JOG_TOLERANCE_CM = 0.25


# ---------------------------------------------------------------- helper processes

def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def start_helpers(log):
    """Start camera capture and the arm server unless they're already running."""
    started = []
    if not running("camera_capture"):
        binary = os.path.join(FRAME_DIR, "camera_capture")
        if not os.path.exists(binary):
            log("building camera_capture ...")
            subprocess.run(["swiftc", "-O", os.path.join(HERE, "camera_capture.swift"), "-o", binary], check=True)
        started.append(subprocess.Popen([binary, FRAME_DIR, TOP, SIDE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        log("started camera capture")
    if not running("arm_server.py"):
        started.append(subprocess.Popen([sys.executable, os.path.join(HERE, "arm_server.py")], cwd=HERE,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        log("started arm server")
        time.sleep(2.5)
    return started


# ---------------------------------------------------------------- vision

class Vision:
    """Latest frame per camera, tub/bagel tracking, knife detection, optional YOLO-World overlay."""

    def __init__(self, use_ai, log):
        refs = st.read_json(st.REFS)
        calibration = st.read_json(st.CALIBRATION) or {}
        self.trackers = {cam: st.CameraTracker(cam, refs["cameras"][cam], calibration) for cam in (TOP, SIDE)}
        self.lock = threading.Lock()
        self.frames, self.states, self.knife = {}, {}, {}
        self.trails = {TOP: [], SIDE: []}  # recent knife-tip pixels, to show the arm's motion
        self.ai = None
        if use_ai:
            try:
                import live_bagel_robot_vision_v2 as v2
                self.v2 = v2
                self.ai_args = SimpleNamespace(conf=0.13, imgsz=416, max_objects=8, surface_res=220,
                                               spread_sensitivity=0.56, near_px=16)
                self.ai = v2.DetectorWorker(self.ai_args)
                log("YOLO-World detector loading (first run downloads weights)")
            except Exception as e:  # noqa: BLE001 - run without the AI overlay rather than not at all
                log(f"AI detector unavailable ({type(e).__name__}: {e}); tracking only")

    def update(self):
        for cam, tracker in self.trackers.items():
            frame = kc.read_frame(cam)
            if frame is None:
                continue
            state = tracker.update(frame)
            taped = st.find_knife_tape(frame)
            with self.lock:
                self.frames[cam], self.states[cam] = frame, state
                self.knife[cam] = taped
                if taped:
                    self.trails[cam] = (self.trails[cam] + [tuple(taped[1])])[-TRAIL_LEN:]
            if cam == TOP and self.ai is not None:
                self.ai.offer(frame)

    def locate(self, name, cam):
        """Locator for knife_center: (box, score) of `name` in `cam`."""
        with self.lock:
            obj = self.states.get(cam, {}).get("objects", {}).get(name)
        return (obj["box"], obj["score"]) if obj else None

    def ready(self, name):
        return all((self.locate(name, cam) or (None, 0))[1] >= kc.MIN_SCORE for cam in (TOP, SIDE))

    def panel(self, cam, label):
        with self.lock:
            frame = self.frames.get(cam)
            state = self.states.get(cam)
            taped = self.knife.get(cam)
            trail = list(self.trails[cam])
        if frame is None:
            out = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
            cv2.putText(out, f"{label}: waiting for camera", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return out
        out = frame.copy()
        if cam == TOP and self.ai is not None:
            objects, updated, _, error = self.ai.read()
            if error:
                cv2.putText(out, "AI: " + error[:60], (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            elif updated and time.monotonic() - updated < 2.0:
                self.v2.draw(out, objects, self.ai_args)
        for name, obj in (state or {}).get("objects", {}).items():
            if obj["score"] < kc.MIN_SCORE:
                continue
            x0, y0, x1, y1 = obj["box"]
            colour = st.COLORS[name]
            cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2)
            cv2.putText(out, f"{name} {obj['score']:.2f}", (x0, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        for i in range(1, len(trail)):  # fading trail of where the knife tip has been
            fade = i / len(trail)
            cv2.line(out, trail[i - 1], trail[i], (0, int(140 * fade), int(255 * fade)), 2)
        if taped:
            base, tip = taped[0], taped[1]
            cv2.line(out, tuple(base), tuple(tip), (0, 0, 255), 3)
            cv2.circle(out, tuple(tip), 6, (0, 0, 255), 2)
            cv2.putText(out, "knife tip", (tip[0] + 8, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(out, label, (8, PANEL_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return cv2.resize(out, (PANEL_W, PANEL_H))

    def close(self):
        if self.ai is not None:
            self.ai.close()


# ---------------------------------------------------------------- arm actions

class Commander:
    """Runs one arm action at a time in the background, so the video never freezes."""

    def __init__(self, vision, log):
        self.vision, self.log = vision, log
        self.jobs = queue.Queue()
        self.stop = threading.Event()
        self.busy = False
        self.step_cm = 1.0      # per jog tick, 0.5 .. 10 cm (slider)
        self.roll_speed = 1.0   # 1 .. 10 x (slider)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            name, fn = self.jobs.get()
            self.busy = True
            self.stop.clear()
            try:
                fn()
            except kc.CenteringError as e:
                self.log(f"{name}: {e}")
            except Exception as e:  # noqa: BLE001 - report and keep the command center alive
                self.log(f"{name} failed: {e}")
            finally:
                self.busy = False

    def submit(self, name, fn, manual=False):
        if self.busy or not self.jobs.empty():
            if not manual:
                self.log(f"busy; press Stop first to start '{name}'")
            return
        self.jobs.put((name, fn))

    # --- automatic
    def to_object(self, target, height_cm, below_cm=0.0):
        def job():
            if height_cm is not None:
                where = f"{height_cm} cm above"
            else:
                where = f"{below_cm} cm below the centre of" if below_cm else "to the centre of"
            self.log(f"moving knife tip {where} the {target} ...")
            kc.center_knife_over(target, height_cm, log=self.log, locate=self.vision.locate, stop=self.stop,
                                 below_cm=below_cm)
        self.submit(f"to {target}", job)

    def to_tub(self):
        self.to_object("tub", None, below_cm=TUB_BELOW_CENTRE_CM)

    def to_bagel(self):
        self.to_object("bagel", None)

    def point_down(self):
        def job():
            state = k.arm_cmd("state")
            joints = {int(a): b for a, b in state["joints"].items()}
            _, tilt = kc.move_tip(np.array(state["points"]["knife_tip"]), joints, 1.0)
            self.log(f"blade {tilt:.1f} deg from vertical")
        self.submit("point down", job)

    def flat(self):
        self.submit("knife flat", lambda: kc.make_knife_flat(log=self.log))

    def halt(self):
        self.stop.set()
        self.log("STOP: halting after the current move")

    # --- manual
    def jog(self, dx=0.0, dy=0.0, dz=0.0):
        """Move the knife tip by (dx, dy, dz) * step cm (each of dx, dy, dz in -1..1),
        keeping the blade's current angle."""
        delta = np.array([dx, dy, dz]) * self.step_cm
        self.submit("jog", lambda: self.move_tip_by(delta), manual=True)

    def move_tip_by(self, delta):
        """Move the knife tip by `delta` (cm), keeping the blade's angle. Runs in the job thread."""
        state = k.arm_cmd("state")
        joints = {int(a): b for a, b in state["joints"].items()}
        base, tip = (np.array(state["points"][p]) for p in ("knife_base", "knife_tip"))
        blade = (tip - base) / np.linalg.norm(tip - base)
        target = tip + np.array(delta, float)
        # Gravity makes the arm fall short (most on the way up), so after each move check
        # where the tip really is and aim at the target again from there.
        aim = target.copy()
        for _ in range(JOG_TRIES):
            sol, err, _ = k.solve_knife(aim, joints, direction=blade, dir_weight=3.0)
            if err > 0.5:
                self.log(f"can't reach that ({err:.1f} cm off); not moving")
                return
            secs = JOG_UP_SECONDS if aim[2] > tip[2] else JOG_SECONDS
            state = k.arm_cmd("joints", joints={str(s): v for s, v in sol.items()}, seconds=secs,
                              speed=JOG_SPEED)
            joints = {int(a): b for a, b in state["joints"].items()}
            tip = np.array(state["points"]["knife_tip"])
            short = target - tip
            if np.linalg.norm(short) < JOG_TOLERANCE_CM:
                return
            aim = aim + short  # overshoot the command by what it fell short

    def roll(self, amount):
        """Roll the hand by `amount` (-1..1) x ROLL_STEP x roll speed."""
        ticks = int(round(amount * ROLL_STEP * self.roll_speed))
        if ticks:
            self.submit("roll", lambda: self.roll_by(ticks), manual=True)

    def roll_by(self, ticks):
        """Turn the wrist roll by `ticks`, within its range. Runs in the job thread. Returns the
        ticks actually turned."""
        joints = k.arm_cmd("state")["joints"]
        start = joints["5"]
        joints["5"] = int(np.clip(start + ticks, 0, 4095))
        k.arm_cmd("joints", joints=joints, seconds=JOG_SECONDS, speed=ROLL_SPEED)
        return joints["5"] - start

    def roll_for(self, seconds):
        """Keep rolling the hand for `seconds` at the roll-speed setting, turning back at the end
        of the roll servo's range. Runs in the job thread."""
        direction, end = 1, time.time() + seconds
        while time.time() < end:
            if self.stop.is_set():
                raise kc.CenteringError("stopped")
            if self.roll_by(direction * int(ROLL_STEP * self.roll_speed)) == 0:
                direction = -direction  # at the end of the range: come back the other way

    # --- sequences
    def tub_to_bagel_sequence(self, roll_seconds=2.0, lift_cm=SEQUENCE_LIFT_CM):
        """Knife to the tub (1 cm below its centre), roll for `roll_seconds`, go up `lift_cm`,
        knife down to the bagel's centre, roll for `roll_seconds` again."""
        def job():
            def step(msg):
                if self.stop.is_set():
                    raise kc.CenteringError("stopped")
                self.log(msg)
            step("sequence 1/5: knife to the tub")
            kc.center_knife_over("tub", None, log=self.log, locate=self.vision.locate, stop=self.stop,
                                 below_cm=TUB_BELOW_CENTRE_CM)
            step(f"sequence 2/5: rolling {roll_seconds:g} s in the tub")
            self.roll_for(roll_seconds)
            step(f"sequence 3/5: up {lift_cm:g} cm")
            self.move_tip_by([0.0, 0.0, lift_cm])
            step("sequence 4/5: knife down to the bagel")
            kc.center_knife_over("bagel", None, log=self.log, locate=self.vision.locate, stop=self.stop)
            step(f"sequence 5/5: rolling {roll_seconds:g} s on the bagel")
            self.roll_for(roll_seconds)
            self.log("sequence done")
        self.submit("tub -> bagel sequence", job)


# ---------------------------------------------------------------- on-screen controls

class Pad:
    """A spring-back control: drag inside it, it reports how far from centre (-1..1 per axis)."""

    def __init__(self, name, x0, y0, x1, y1, axes):
        self.name, self.rect, self.axes = name, (x0, y0, x1, y1), axes  # axes: "xy", "y" or "x"
        self.value = (0.0, 0.0)
        self.held = False

    def contains(self, x, y):
        x0, y0, x1, y1 = self.rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def drag(self, x, y):
        x0, y0, x1, y1 = self.rect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        vx = np.clip((x - cx) / ((x1 - x0) / 2), -1, 1) if "x" in self.axes else 0.0
        vy = np.clip((cy - y) / ((y1 - y0) / 2), -1, 1) if "y" in self.axes else 0.0  # up = +
        if self.axes == "xy" and np.hypot(vx, vy) > 1:
            vx, vy = vx / np.hypot(vx, vy), vy / np.hypot(vx, vy)
        self.value = (float(vx), float(vy))

    def release(self):
        self.held, self.value = False, (0.0, 0.0)

    def draw(self, canvas):
        x0, y0, x1, y1 = self.rect
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if self.axes == "xy":
            r = (x1 - x0) // 2
            cv2.circle(canvas, (cx, cy), r, (90, 90, 90), -1)
            cv2.circle(canvas, (cx, cy), r, (200, 200, 200), 1)
            cv2.line(canvas, (cx - r, cy), (cx + r, cy), (130, 130, 130), 1)
            cv2.line(canvas, (cx, cy - r), (cx, cy + r), (130, 130, 130), 1)
        else:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (90, 90, 90), -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)
        kx = int(cx + self.value[0] * (x1 - x0) / 2)
        ky = int(cy - self.value[1] * (y1 - y0) / 2)
        cv2.circle(canvas, (kx, ky), 14, (0, 200, 255) if self.held else (220, 220, 220), -1)
        cv2.putText(canvas, self.name, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


class Slider:
    """A setting slider: drag to set a value between lo and hi."""

    def __init__(self, name, x0, y0, x1, y1, lo, hi, value, fmt, on_change):
        self.name, self.rect, self.lo, self.hi, self.value = name, (x0, y0, x1, y1), lo, hi, value
        self.fmt, self.on_change, self.held = fmt, on_change, False
        on_change(value)

    def contains(self, x, y):
        x0, y0, x1, y1 = self.rect
        return x0 <= x <= x1 and y0 - 8 <= y <= y1 + 8

    def drag(self, x, _y):
        x0, _, x1, _ = self.rect
        f = float(np.clip((x - x0) / (x1 - x0), 0, 1))
        self.value = round((self.lo + f * (self.hi - self.lo)) * 2) / 2  # 0.5 steps
        self.on_change(self.value)

    def release(self):
        self.held = False

    def draw(self, canvas):
        x0, y0, x1, y1 = self.rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (90, 90, 90), -1)
        f = (self.value - self.lo) / (self.hi - self.lo)
        cv2.rectangle(canvas, (x0, y0), (int(x0 + f * (x1 - x0)), y1), (0, 140, 255), -1)
        cv2.circle(canvas, (int(x0 + f * (x1 - x0)), (y0 + y1) // 2), 10, (230, 230, 230), -1)
        cv2.putText(canvas, f"{self.name}: {self.fmt.format(self.value)}", (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


class Controls:
    """On-screen joystick, sliders and buttons, the keyboard and a game controller - all mapped to
    the same actions. Holding a joystick/slider keeps the knife moving, one tick at a time."""

    def __init__(self, cmd):
        self.cmd = cmd
        y = PANEL_H + 40
        self.xy = Pad("forward/back + left/right", 20, y, 200, y + 180, "xy")
        self.z = Pad("up/down", 225, y, 275, y + 180, "y")
        self.rollpad = Pad("roll", 300, y, 560, y + 40, "x")
        self.pads = [self.xy, self.z, self.rollpad]
        self.sliders = [
            Slider("step per tick", 300, y + 85, 560, y + 105, 0.5, 10.0, 1.0, "{:.1f} cm",
                   lambda v: setattr(cmd, "step_cm", v)),
            Slider("roll speed", 300, y + 150, 560, y + 170, 1.0, 10.0, 1.0, "{:.1f}x",
                   lambda v: setattr(cmd, "roll_speed", v)),
        ]
        self.buttons = []  # (label, x0, y0, x1, y1, action)
        for i, (label, action) in enumerate([("Knife -> tub", cmd.to_tub), ("Knife -> bagel", cmd.to_bagel),
                                             ("Point down", cmd.point_down), ("Knife flat", cmd.flat),
                                             ("Tub -> bagel", cmd.tub_to_bagel_sequence), ("STOP", cmd.halt)]):
            x0 = 590 + (i % 3) * 228
            y0 = y + (i // 3) * 52
            self.buttons.append((label, x0, y0, x0 + 220, y0 + 44, action))
        self.active = None
        self.last_tick = 0.0
        self.keys = {ord("w"): lambda: cmd.jog(dx=1), ord("s"): lambda: cmd.jog(dx=-1),
                     ord("a"): lambda: cmd.jog(dy=1), ord("d"): lambda: cmd.jog(dy=-1),
                     ord("r"): lambda: cmd.jog(dz=1), ord("f"): lambda: cmd.jog(dz=-1),
                     ord("z"): lambda: cmd.roll(-1), ord("c"): lambda: cmd.roll(1),
                     ord("t"): cmd.to_tub, ord("b"): cmd.to_bagel,
                     ord("v"): cmd.point_down, ord(" "): cmd.halt, ord("g"): cmd.tub_to_bagel_sequence,
                     ord("h"): cmd.flat,
                     ord("1"): lambda: self.set_step(0.5), ord("2"): lambda: self.set_step(1.0),
                     ord("3"): lambda: self.set_step(2.0), ord("4"): lambda: self.set_step(5.0),
                     ord("5"): lambda: self.set_step(10.0)}
        self.joystick = None
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count():
                self.pg = pygame
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
        except Exception:  # noqa: BLE001 - no pygame or no controller: on-screen controls still work
            self.joystick = None

    def set_step(self, cm):
        self.sliders[0].value = cm
        self.cmd.step_cm = cm
        self.cmd.log(f"step {cm} cm")

    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            for label, x0, y0, x1, y1, action in self.buttons:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    action()
                    return
            for ctl in self.pads + self.sliders:
                if ctl.contains(x, y):
                    self.active, ctl.held = ctl, True
                    ctl.drag(x, y)
                    return
        elif event == cv2.EVENT_MOUSEMOVE and self.active is not None:
            self.active.drag(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.active is not None:
            self.active.release()
            self.active = None

    def on_key(self, key):
        action = self.keys.get(key)
        if action:
            action()

    def tick(self):
        """While a joystick/slider is held, keep sending moves scaled by how far it's pushed."""
        if self.cmd.busy or time.time() - self.last_tick < TICK_SECONDS:
            return
        vx, vy = self.xy.value
        vz = self.z.value[1]
        vr = self.rollpad.value[0]
        sticks = self._controller_axes()
        if sticks:
            vx, vy, vz, vr = (a if abs(a) > abs(b) else b for a, b in zip((vx, vy, vz, vr), sticks))
        dead = lambda v: v if abs(v) > PAD_DEADZONE else 0.0
        # On the pad, up = forward (away from the robot) and right = robot's right (-y).
        dx, dy, dz, dr = dead(vy), dead(-vx), dead(vz), dead(vr)
        if dx or dy or dz:
            self.cmd.jog(dx, dy, dz)
            self.last_tick = time.time()
        elif dr:
            self.cmd.roll(dr)
            self.last_tick = time.time()

    def _controller_axes(self):
        """Game controller (Xbox/PlayStation layout): left stick moves, right stick Y up/down,
        triggers roll; buttons A tub, B bagel, X point down, Y stop."""
        if self.joystick is None:
            return None
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.JOYBUTTONDOWN:
                {0: self.cmd.to_tub, 1: self.cmd.to_bagel, 2: self.cmd.point_down,
                 3: self.cmd.halt}.get(event.button, lambda: None)()
        axes = [self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())] + [0.0] * 6
        roll = 0.0
        if self.joystick.get_numbuttons() > 5:
            roll = float(self.joystick.get_button(5)) - float(self.joystick.get_button(4))  # RB / LB
        return axes[0], -axes[1], -axes[3], roll

    def draw(self, canvas, busy):
        for ctl in self.pads + self.sliders:
            ctl.draw(canvas)
        for label, x0, y0, x1, y1, _ in self.buttons:
            colour = (40, 40, 200) if label == "STOP" else (70, 70, 70)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)
            cv2.putText(canvas, label, (x0 + 10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        pad = "controller connected" if self.joystick is not None else "no controller"
        cv2.putText(canvas, f"{'MOVING' if busy else 'ready'} | {pad} | keys: WASD RF ZC T B G V H space 1-5",
                    (590, PANEL_H + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def draw_telemetry(canvas, x, y):
    """Live arm position from arm_server.py (updated ~10x a second, including during moves)."""
    state = st.read_json(st.KNIFE_STATE)
    if not state or "points" not in state:
        cv2.putText(canvas, "arm position: waiting for arm_server", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return
    tip = state["points"].get("knife_tip")
    base = state["points"].get("knife_base")
    age = time.time() - state["time"]
    lines = []
    if tip:
        lines.append(f"knife tip  x {tip[0]:6.1f}  y {tip[1]:6.1f}  z {tip[2]:6.1f} cm")
    if tip and base:
        blade = np.array(tip) - np.array(base)
        tilt = np.degrees(np.arccos(np.clip(-blade[2] / np.linalg.norm(blade), -1, 1)))
        lines.append(f"blade {tilt:5.1f} deg from straight down   wrist pitch {state['pitch']:.0f} deg")
    j = state["joints"]
    lines.append("joints  " + "  ".join(f"{n} {j[str(i)]}" for i, n in
                                        ((1, "base"), (2, "shldr"), (3, "elbow"), (4, "wrist"), (5, "roll"), (6, "grip"))))
    colour = (0, 255, 0) if age < 0.5 else (0, 165, 255)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (x, y + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


# ---------------------------------------------------------------- main loop

def main():
    parser = argparse.ArgumentParser(description="Robot command center")
    parser.add_argument("--no-start", action="store_true", help="don't move the knife to the tub on start-up")
    parser.add_argument("--no-ai", action="store_true", help="skip the YOLO-World detector")
    parser.add_argument("--start-height", type=float, default=1.0, help="cm above the tub on start-up")
    args = parser.parse_args()

    messages = []

    def log(msg):
        print(msg, flush=True)
        messages.append(msg)
        del messages[:-5]

    started = start_helpers(log)
    vision = Vision(not args.no_ai, log)
    cmd = Commander(vision, log)
    controls = Controls(cmd)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, controls.on_mouse)
    start_at = None if args.no_start else time.time() + 3.0  # 3 s to step back before the first move

    try:
        while True:
            vision.update()
            if start_at and time.time() >= start_at:
                if vision.ready("tub"):
                    cmd.to_object("tub", args.start_height)
                    start_at = None
                elif time.time() - start_at > 20:
                    log("start-up: tub not seen in both cameras; use 'Knife -> tub' when it is")
                    start_at = None
            canvas = np.zeros((PANEL_H + BAR_H, 2 * PANEL_W, 3), np.uint8)
            canvas[:PANEL_H, :PANEL_W] = vision.panel(TOP, "top view")
            canvas[:PANEL_H, PANEL_W:] = vision.panel(SIDE, "side view (height)")
            if start_at:
                cv2.putText(canvas, f"moving knife to tub in {max(0, start_at - time.time()):.0f} s - stand clear",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            for i, msg in enumerate(messages):
                cv2.putText(canvas, msg[:110], (PANEL_W + 8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            controls.draw(canvas, cmd.busy)
            draw_telemetry(canvas, 590, PANEL_H + 180)
            cv2.imshow(WINDOW, canvas)
            controls.tick()
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q")):
                break
            if key != 255:
                controls.on_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        cmd.halt()
        vision.close()
        cv2.destroyAllWindows()
        for p in started:  # leave helpers we didn't start alone
            p.terminate()


if __name__ == "__main__":
    main()

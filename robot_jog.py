# Small, guarded moves on one arm plus a snapshot from the robot camera, for step-by-step control.
#
#   python3 robot_jog.py 18911                      # just hold the pose and take a snapshot
#   python3 robot_jog.py 18911 2=-50 3=+80          # move shoulder -50 ticks, elbow +80 ticks
#   python3 robot_jog.py 18911 6=open               # gripper open / close / an absolute position
#   python3 robot_jog.py 18911 release              # go limp (support the arm first!)
#
# The arm is left powered and holding its pose when the script exits. Every move uses the same
# contact guard and torque caps as robot_up_down.py.
import argparse
import os
import sys
import time

import robot_up_down as r

SNAP_DIR = os.environ.get("SNAP_DIR", os.path.dirname(os.path.abspath(__file__)))


def snapshot(path):
    """Grab a frame from the USB robot camera (not the MacBook's 1080p camera), upright."""
    import cv2
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 1920:
            cap.release()
            continue
        frame, t = None, time.time()
        while time.time() - t < 5:
            ok, f = cap.read()
            if ok:
                frame = f
                if time.time() - t > 1.0:  # let exposure settle
                    break
            time.sleep(0.03)
        cap.release()
        if frame is not None:
            cv2.imwrite(path, cv2.rotate(frame, cv2.ROTATE_180))  # camera is mounted upside down
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", help="last digits of the arm's port, e.g. 18911")
    parser.add_argument("moves", nargs="*", help="joint=delta, joint=open|close|=ABS, or 'release'")
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--snap", default="snap.jpg")
    args = parser.parse_args()

    port = next((p for p in sorted(r.glob.glob(r.PORT_PATTERN)) if p.endswith(args.arm)), None)
    if not port:
        sys.exit(f"no arm on a port ending {args.arm}")
    arm = r.Arm(port)
    try:
        pose = arm.hold()
        if args.moves == ["release"]:
            arm.release()
            print("released")
            return
        target = dict(pose)
        for m in args.moves:
            sid, val = m.split("=", 1)
            sid = int(sid)
            if val in ("open", "close"):
                lo, hi = arm.limits(sid)
                target[sid] = round(lo + 0.75 * (hi - lo)) if val == "open" else lo
            elif val.startswith("="):
                target[sid] = int(val[1:])
            else:
                target[sid] = pose[sid] + int(val)
            lo, hi = arm.limits(sid)
            target[sid] = max(lo, min(hi, target[sid]))
        if target != pose:
            try:
                r.move_all([arm], [target], args.seconds)
            except r.Contact as e:
                arm.freeze()
                print(f"CONTACT: {e} -- stopped there")
            time.sleep(0.4)
        state = {sid: arm.read_state(sid) for sid in r.ALL_JOINTS}
        print("pose:", {sid: p for sid, (p, _) in state.items()})
        print("load:", {sid: l for sid, (_, l) in state.items()})
    finally:
        arm.close()
    path = snapshot(os.path.join(SNAP_DIR, args.snap))
    print("snapshot:", path or "camera gave no frame")


if __name__ == "__main__":
    main()

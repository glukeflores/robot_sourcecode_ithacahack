# Live view of the robot cameras, side by side.
#
#   python3 camera_view.py [SAVE_DIR]       # press q in the window (or Ctrl+C) to quit
#
# camera_capture (built from camera_capture.swift) does the capturing and keeps the latest frame
# of each camera in SAVE_DIR/live_<uniqueID>.jpg; this just shows those files in one window.
# (Capturing with OpenCV doesn't work here: it can't stream two of these cameras through one hub.)
import glob
import os
import sys

import cv2
import numpy as np

SAVE_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
LABELS = {
    "0x1110000328f0073": "overview",
    "0x11300001e458022": "arm 18911 (no knife)",
    "0x11200001e458022": "arm 30541 (knife), upside down",
}
UPSIDE_DOWN = {"0x11200001e458022"}


def main():
    try:
        while True:
            tiles = []
            for path in sorted(glob.glob(os.path.join(SAVE_DIR, "live_*.jpg"))):
                cam = os.path.basename(path)[5:-4]
                frame = cv2.imread(path)
                if frame is None:
                    continue
                if cam in UPSIDE_DOWN:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                tile = cv2.resize(frame, (640, 480))
                cv2.putText(tile, LABELS.get(cam, cam), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                tiles.append(tile)
            cv2.imshow("Robot cameras (q to quit)", np.hstack(tiles) if tiles else np.zeros((480, 640, 3), np.uint8))
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

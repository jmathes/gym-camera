#!/usr/bin/env python3
"""gym_loop — presence -> plugs -> Beeminder, on top of gym.py.

    gym_loop.py run [--dry-run] [--once]

Loop:
  - poll the camera's own person/vehicle/pet detector every POLL_INTERVAL
    seconds (cheap: one HTTP call, no local ML).
  - person seen and no session active -> start a session, turn on both plugs
    (fan + Aromadd).
  - during an active session, every FACE_CHECK_INTERVAL seconds: snap a
    still, check it against the reference photos in local/faces/joe/*.jpg. First
    match logs the gymvisit Beeminder datapoint for this session (once).
  - no person seen for SESSION_GAP seconds -> end the session, turn both
    plugs off. A brief step out of frame doesn't end the session early.

Needs the `gym` conda env (onnxruntime, opencv-python-headless,
face_recognition) and REOLINK_USER/REOLINK_PASSWORD/BEEMINDER_AUTH_TOKEN/
BEEMINDER_USER in ~/.local_bashrc, same as gym.py.

Put a handful of clear, front-on photos of yourself from this camera's angle
in local/faces/joe/ before relying on the face-match step — with none present the
loop still runs (presence -> plugs works), it just never matches anyone, so
nothing gets logged to Beeminder.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import face_recognition
import numpy as np

from gym import CAMERA_HOST, Camera, Plug, PLUGS, _bashrc_exports
import os

POLL_INTERVAL = 5          # seconds between "is anyone there" checks
FACE_CHECK_INTERVAL = 30   # seconds between face-match attempts during a session
SESSION_GAP = 15 * 60      # seconds of "nobody" before a session is considered over
FACE_TOLERANCE = 0.6       # face_recognition's default; lower = stricter

FACES_DIR = Path(__file__).parent / "local" / "faces" / "joe"
GYM_GOAL = "gymvisit"
BEEMINDER_API = "https://www.beeminder.com/api/v1"
NTFY_TOPIC = "REDACTED-NTFY-TOPIC"  # private/unguessable; subscribed on Joe's phone in the ntfy app


def notify(message: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}",
                                                        data=message.encode()), timeout=10)
    except urllib.error.URLError as e:
        print(f"notify failed: {e}", file=sys.stderr)


def load_known_faces(faces_dir: Path) -> list[np.ndarray]:
    encodings = []
    for path in sorted(faces_dir.glob("*.jpg")) + sorted(faces_dir.glob("*.png")):
        image = face_recognition.load_image_file(path)
        found = face_recognition.face_encodings(image)
        if found:
            encodings.append(found[0])
        else:
            print(f"warning: no face found in {path}, skipping", file=sys.stderr)
    return encodings


def matches_joe(jpeg_bytes: bytes, known: list[np.ndarray]) -> bool:
    if not known:
        return False
    image = face_recognition.load_image_file(io.BytesIO(jpeg_bytes))
    for encoding in face_recognition.face_encodings(image):
        if any(face_recognition.compare_faces(known, encoding, tolerance=FACE_TOLERANCE)):
            return True
    return False


def log_gymvisit(comment: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would log {GYM_GOAL}: {comment}")
        return
    env = {**_bashrc_exports(), **os.environ}
    token, user = env.get("BEEMINDER_AUTH_TOKEN"), env.get("BEEMINDER_USER")
    if not token or not user:
        print("BEEMINDER_AUTH_TOKEN/BEEMINDER_USER not set, can't log", file=sys.stderr)
        return
    url = f"{BEEMINDER_API}/users/{user}/goals/{GYM_GOAL}/datapoints.json"
    form = urllib.parse.urlencode({"auth_token": token, "value": 1, "comment": comment}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, form), timeout=30) as r:
            r.read()
        print(f"logged {GYM_GOAL}: {comment}")
    except urllib.error.URLError as e:
        print(f"failed to log {GYM_GOAL}: {e}", file=sys.stderr)


@dataclass
class Session:
    active: bool = False
    last_seen: float = 0.0
    last_face_check: float = 0.0
    matched: bool = False

    def start(self) -> None:
        self.active, self.matched = True, False
        print("session start: turning on fan + Aromadd")
        for key in PLUGS:
            Plug(key, PLUGS[key]).set(True)

    def end(self, dry_run: bool) -> None:
        print("session end: turning off fan + Aromadd")
        for key in PLUGS:
            Plug(key, PLUGS[key]).set(False)
        if self.matched:
            log_gymvisit("gym camera auto-detect", dry_run)
        self.active = False


def run(dry_run: bool, once: bool) -> None:
    camera = Camera.from_env(CAMERA_HOST)
    known = load_known_faces(FACES_DIR)
    print(f"{len(known)} reference face(s) loaded from {FACES_DIR}")
    session = Session()

    while True:
        now = time.time()
        present = camera.sees_person()

        if present:
            if not session.active:
                session.start()
            session.last_seen = now
            if not session.matched and now - session.last_face_check >= FACE_CHECK_INTERVAL:
                session.last_face_check = now
                if matches_joe(camera.snap(), known):
                    session.matched = True
                    print("face match: Joe confirmed for this session")
                    notify("Gym camera recognized you — this session will count.")
        elif session.active and now - session.last_seen >= SESSION_GAP:
            session.end(dry_run)

        if once:
            break
        time.sleep(POLL_INTERVAL)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the presence/plugs/Beeminder loop")
    r.add_argument("--dry-run", action="store_true", help="print what would be logged instead of calling Beeminder")
    r.add_argument("--once", action="store_true", help="one poll iteration, then exit (for testing)")
    a = p.parse_args()
    run(a.dry_run, a.once)


if __name__ == "__main__":
    main()

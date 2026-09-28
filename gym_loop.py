#!/usr/bin/env python3
"""gym_loop — presence -> plugs -> Beeminder, on top of gym.py.

    gym_loop.py run [--dry-run] [--once]

Two independent modes, on purpose -- presence is reliable and identity isn't,
so identity failing should never take the plugs down with it:

  - Presence mode: every POLL_INTERVAL seconds, ask the camera's own
    person/vehicle/pet detector whether anyone's there (cheap: one HTTP call,
    no local ML). If so, turn both plugs on with a PLUG_TIMEOUT auto-off
    timer set on the Shelly's own firmware, refreshed every poll. There's no
    software "off" command anymore -- the plug turns itself off on schedule
    even if this whole process dies, which is the point.
  - Identity mode: while a presence session is active, every
    FACE_CHECK_INTERVAL seconds (~1Hz): snap a still, archive it (see below),
    and -- until matched once this session -- check it against the reference
    photos in local/faces/joe/*.jpg. Matching is a two-stage locate-cheap/
    encode-precise split (see matches_joe) so this stays affordable at ~1Hz
    instead of the ~3s a naive full-res check costs. The moment a match
    lands: a phone notification fires, and the Beeminder datapoint is
    created immediately too (value=1, a placeholder comment) so credit shows
    up right away instead of after a 15-minute wait.
  - A presence session ends after SESSION_GAP seconds of nobody seen (a
    brief step out of frame doesn't split one visit into two). That's when,
    if matched, the same Beeminder datapoint gets PUT-updated in place with
    its final comment, adding duration measured from first-seen to last-seen
    (not counting the SESSION_GAP wait itself) -- the value was already
    right, only the comment changes. This gap timer is bookkeeping for
    Beeminder attribution only -- it doesn't gate the plugs, which run on
    their own hardware timer.

Logs and the photo archive live under ~/.local/state/gym-camera/ (XDG state
dir — runtime data, not config, not backed up), not in the repo:
  - logs/gym-camera.log — rotating log of session/match/Beeminder events.
  - photos/YYYYMMDD-HHMMSS.jpg — one still per FACE_CHECK_INTERVAL for every
    second someone's present, pruned after PHOTO_RETENTION_DAYS.

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
import json
import logging
import logging.handlers
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image

from gym import CAMERA_HOST, Camera, Plug, PLUGS, _bashrc_exports
import os

POLL_INTERVAL = 1          # seconds between "is anyone there" checks (cheap: ~0.07s/call)
FACE_CHECK_INTERVAL = 1    # seconds between face-match attempts during a session -- feasible
                           # at ~1Hz because of the two-stage locate/encode split below, not
                           # because full-res detection got any cheaper (it's still ~3s alone)
SESSION_GAP = 15 * 60      # seconds of "nobody" before a Beeminder-attribution session is over
FACE_TOLERANCE = 0.6       # face_recognition's default; lower = stricter
FACE_LOCATE_MAX_DIM = 640  # downscale target for the cheap first-pass face *location* search
PHOTO_RETENTION_DAYS = 7   # how long to keep archived presence photos
PLUG_TIMEOUT = 10 * 60     # seconds; refreshed on every poll while present, so the Shelly
                           # itself turns the plugs off if this process dies or hangs

FACES_DIR = Path(__file__).parent / "local" / "faces" / "joe"
GYM_GOAL = "gymvisit"
BEEMINDER_API = "https://www.beeminder.com/api/v1"
NTFY_TOPIC = "REDACTED-NTFY-TOPIC"  # private/unguessable; subscribed on Joe's phone in the ntfy app

# XDG state dir: logs and the rolling photo archive are runtime state, not
# config and not something to back up, so they don't belong in the repo.
STATE_DIR = Path.home() / ".local" / "state" / "gym-camera"
LOG_DIR = STATE_DIR / "logs"
PHOTOS_DIR = STATE_DIR / "photos"

log = logging.getLogger("gym_loop")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "gym-camera.log", maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)


def archive_photo(jpeg_bytes: bytes) -> None:
    """Save a presence-time photo and prune anything older than the retention window."""
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    (PHOTOS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.jpg").write_bytes(jpeg_bytes)
    cutoff = time.time() - PHOTO_RETENTION_DAYS * 86400
    for path in PHOTOS_DIR.glob("*.jpg"):
        if path.stat().st_mtime < cutoff:
            path.unlink()


def notify(message: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}",
                                                        data=message.encode()), timeout=10)
    except urllib.error.URLError as e:
        log.warning(f"notify failed: {e}")


def load_known_faces(faces_dir: Path) -> list[np.ndarray]:
    encodings = []
    for path in sorted(faces_dir.glob("*.jpg")) + sorted(faces_dir.glob("*.png")):
        image = face_recognition.load_image_file(path)
        found = face_recognition.face_encodings(image)
        if found:
            encodings.append(found[0])
        else:
            log.warning(f"no face found in {path}, skipping")
    return encodings


def matches_joe(jpeg_bytes: bytes, known: list[np.ndarray]) -> bool:
    """Two-stage: locate faces on a cheap downscaled copy (~0.2s regardless of the source
    resolution), then encode only those regions from the original full-res image (~0.15s per
    face). This is what makes ~1Hz checking affordable -- running detection on the full
    2560x1920 frame directly costs ~3s on its own, every time.

    Logs every attempt, not just successes -- a check that ran and found no usable face is a
    different, useful fact from one that found and matched a face, and both are different from
    this process not running at all."""
    full = face_recognition.load_image_file(io.BytesIO(jpeg_bytes))
    h, w = full.shape[:2]
    scale = min(1.0, FACE_LOCATE_MAX_DIM / max(h, w))
    small = np.array(Image.fromarray(full).resize((int(w * scale), int(h * scale)))) if scale < 1.0 else full

    locations = face_recognition.face_locations(small)
    if not locations:
        log.info("face check: no face detected in frame")
        return False

    inv = 1 / scale
    full_locations = [(int(t * inv), int(r * inv), int(b * inv), int(l * inv)) for t, r, b, l in locations]
    faces = face_recognition.face_encodings(full, known_face_locations=full_locations)

    if not known:
        log.info(f"face check: {len(faces)} face(s) detected, but no reference photos loaded")
        return False
    for encoding in faces:
        if any(face_recognition.compare_faces(known, encoding, tolerance=FACE_TOLERANCE)):
            log.info(f"face check: {len(faces)} face(s) detected, matched Joe")
            return True
    log.info(f"face check: {len(faces)} face(s) detected, none matched Joe")
    return False


def _beeminder_auth() -> tuple[str, str] | None:
    env = {**_bashrc_exports(), **os.environ}
    token, user = env.get("BEEMINDER_AUTH_TOKEN"), env.get("BEEMINDER_USER")
    if not token or not user:
        log.error("BEEMINDER_AUTH_TOKEN/BEEMINDER_USER not set, can't log")
        return None
    return token, user


def log_gymvisit(comment: str, dry_run: bool) -> str | None:
    """Creates the datapoint immediately on match, before the session's duration is known.
    Returns the new datapoint's id so update_gymvisit_duration can fill that in once it is."""
    if dry_run:
        log.info(f"[dry-run] would log {GYM_GOAL}: {comment}")
        return None
    auth = _beeminder_auth()
    if not auth:
        return None
    token, user = auth
    url = f"{BEEMINDER_API}/users/{user}/goals/{GYM_GOAL}/datapoints.json"
    form = urllib.parse.urlencode({"auth_token": token, "value": 1, "comment": comment}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, form), timeout=30) as r:
            point = json.loads(r.read())
        log.info(f"logged {GYM_GOAL}: {comment} (id {point['id']})")
        return point["id"]
    except urllib.error.URLError as e:
        log.error(f"failed to log {GYM_GOAL}: {e}")
        return None


def update_gymvisit_duration(datapoint_id: str | None, comment: str, dry_run: bool) -> None:
    if dry_run:
        log.info(f"[dry-run] would update {GYM_GOAL} datapoint: {comment}")
        return
    if not datapoint_id:
        return
    auth = _beeminder_auth()
    if not auth:
        return
    token, user = auth
    url = f"{BEEMINDER_API}/users/{user}/goals/{GYM_GOAL}/datapoints/{datapoint_id}.json"
    form = urllib.parse.urlencode({"auth_token": token, "comment": comment}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, form, method="PUT"), timeout=30) as r:
            r.read()
        log.info(f"updated {GYM_GOAL} datapoint {datapoint_id}: {comment}")
    except urllib.error.URLError as e:
        log.error(f"failed to update {GYM_GOAL} datapoint {datapoint_id}: {e}")


def refresh_plugs() -> None:
    """Turn both plugs on and (re)arm their own PLUG_TIMEOUT auto-off, on the Shelly itself.
    Called on every poll where a person is present, regardless of whose face (if anyone's)
    that turns out to be — this is the presence mode, not the identity mode. Because the
    timer lives on the plug's own firmware, the fan and Aromadd still turn themselves off
    within PLUG_TIMEOUT even if this whole process crashes or the network drops."""
    for key, host in PLUGS.items():
        Plug(key, host).set(True, toggle_after=PLUG_TIMEOUT)


@dataclass
class Session:
    """Tracks one continuous presence for Beeminder attribution. Doesn't touch the plugs —
    that's refresh_plugs()'s job, on presence alone, independent of identity."""
    active: bool = False
    started_at: float = 0.0
    last_seen: float = 0.0
    last_face_check: float = 0.0
    matched: bool = False
    datapoint_id: str | None = None

    def start(self) -> None:
        self.active, self.matched, self.datapoint_id = True, False, None
        self.started_at = time.time()
        log.info("session start")

    def end(self, dry_run: bool) -> None:
        # last_seen, not now: duration is first-seen to last-seen, not including the
        # trailing SESSION_GAP wait before we're confident you've actually left.
        minutes = (self.last_seen - self.started_at) / 60
        log.info(f"session end ({minutes:.0f} min)")
        if self.matched:
            update_gymvisit_duration(self.datapoint_id, f"gym camera auto-detect, ~{minutes:.0f} min", dry_run)
        self.active = False


def run(dry_run: bool, once: bool) -> None:
    setup_logging()
    camera = Camera.from_env(CAMERA_HOST)
    known = load_known_faces(FACES_DIR)
    log.info(f"{len(known)} reference face(s) loaded from {FACES_DIR}")
    session = Session()

    while True:
        now = time.time()
        present = camera.sees_person()

        if present:
            refresh_plugs()  # presence mode: on/refresh regardless of identity

            if not session.active:
                session.start()
            session.last_seen = now
            # Snap + archive on this cadence regardless of match state, so the
            # photo archive covers the whole session, not just until the first
            # match. Face-matching itself stops once matched, since there's
            # nothing left to determine.
            if now - session.last_face_check >= FACE_CHECK_INTERVAL:
                session.last_face_check = now
                photo = camera.snap()
                archive_photo(photo)
                if not session.matched and matches_joe(photo, known):
                    session.matched = True
                    notify("Gym camera recognized you — this session will count.")
                    session.datapoint_id = log_gymvisit("gym camera auto-detect (session in progress)", dry_run)
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

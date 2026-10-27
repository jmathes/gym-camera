#!/usr/bin/env python3
"""gym_loop — presence -> plugs -> Beeminder, on top of gym.py.

    gym_loop.py presence             one presence cycle: poll cameras, log, refresh plugs
    gym_loop.py tick [--dry-run]     one identity cycle: session, face check, Beeminder

Each invocation runs exactly one cycle and exits. The loop comes from outside,
from two systemd user units, so slow face work can never delay presence:
  - gym-presence.timer starts `presence` on every wall-clock second. It never
    imports the face libraries, so it finishes in ~0.25s with both cameras up.
  - gym-camera.service has Restart=always, so systemd starts the next `tick`
    a second after the last one exits, whether it exited cleanly or crashed.
    Ticks read presence from PRESENCE_FILE rather than polling cameras.
Each command is the only writer of its own state file (PRESENCE_FILE,
STATE_FILE). A cycle that crashes partway just gets rerun from the last saved
state, and every step is safe to repeat: plug refreshes re-arm the same timer,
and log_gymvisit finds today's datapoint before creating one.

A camera being offline is a normal condition inside a cycle, not an error:
it counts as not seeing anyone, a face check that lands on it is skipped, and
its going offline/online is logged once per transition.

Two independent modes, on purpose -- presence is reliable and identity isn't,
so identity failing should never take the plugs down with it:

  - Presence mode: every second, ask each camera's own person/vehicle/pet
    detector whether anyone's there (cheap: one HTTP call per camera, no
    local ML) and append the answer to the presence log. If either camera
    sees someone, turn both plugs on with a PLUG_TIMEOUT auto-off timer set
    on the Shelly's own firmware, refreshed every second. There's no
    software "off" command anymore -- the plug turns itself off on schedule
    even if this whole process dies, which is the point.
  - Identity mode: while a presence session is active and not yet matched,
    at most every FACE_CHECK_INTERVAL seconds: snap a still from the next
    camera in rotation, archive it (see
    below), and check it against the reference photos in local/faces/joe/*.jpg.
    Matching is a two-stage locate-cheap/encode-precise split (see
    matches_joe) so each check costs well under a second instead of the ~3s
    a naive full-res check costs. A tick that does a face check also pays
    ~2s to import face_recognition, so in practice checks run every ~3s
    while unmatched. Alternating keeps total snapshot traffic the same as
    one camera had alone. The rotation doesn't skip a camera whose own
    detector sees nobody: face detection might find a face that the
    camera's person detector missed. The moment a match lands: a phone notification
    fires, and the Beeminder datapoint is created immediately too (value=1,
    a placeholder comment) so credit shows up right away instead of after a
    15-minute wait. Once matched, snapshotting drops to ARCHIVE_INTERVAL for
    the rest of the session -- there's no more identity to determine, and
    each full-res snap is a real ~400KB pull over the camera's WiFi link, not
    something to keep doing every second for an hour-long workout.
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
  - logs/gym-camera.log — rotating log of presence changes, camera
    offline/online, session/match/Beeminder events.
  - presence/YYYYMMDD-HH.tsv — one line per second: time and each camera's
    answer (person / nobody / offline). Pruned after PRESENCE_RETENTION_HOURS.
  - presence.json — the latest presence, written by `presence`, read by `tick`.
  - state.json — what one tick hands to the next.
  - known_faces.npz — cached encodings of local/faces/joe/, rebuilt when
    those files change (encoding them from scratch takes minutes).
  - photos/YYYYMMDD-HHMMSS-CAMERA.jpg — one still per FACE_CHECK_INTERVAL while
    unmatched, then one per ARCHIVE_INTERVAL for the rest of the session,
    pruned after PHOTO_RETENTION_DAYS.

Needs the `gym` conda env (onnxruntime, opencv-python-headless,
face_recognition) and REOLINK_USER/REOLINK_PASSWORD/BEEMINDER_AUTH_TOKEN/
BEEMINDER_USER in ~/.local_bashrc, same as gym.py.

Put a handful of clear, front-on photos of yourself from both cameras' angles
in local/faces/joe/ before relying on the face-match step — with none present
the loop still runs (presence -> plugs works), it just never matches anyone, so
nothing gets logged to Beeminder.
"""
from __future__ import annotations

import argparse
import http.client
import io
import json
import logging
import logging.handlers
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from gym import CAMERAS, Camera, GYM_PLUGS, Plug, _bashrc_exports

# face_recognition, numpy and PIL take ~2s to import, which would dominate a ~0.1s
# presence-only cycle, so they're imported inside the functions that need them.
if TYPE_CHECKING:
    import numpy as np

PRESENCE_TIMEOUT = 0.8     # seconds to wait on a camera's person check (or a plug refresh)
                           # before calling it offline for this second -- a healthy one answers
                           # in ~0.07s, and the presence cycle has to finish inside its second
FACE_CHECK_INTERVAL = 1    # seconds between face-match attempts, only while still unmatched --
                           # feasible at ~1Hz because of the two-stage locate/encode split below,
                           # not because full-res detection got any cheaper (still ~3s alone)
ARCHIVE_INTERVAL = 30      # seconds between archive-only snapshots once already matched -- each
                           # full-res photo is a real ~400KB pull over the camera's WiFi link, so
                           # once identity's settled there's no reason to keep pulling one a
                           # second for the rest of a session; that cost is what caused
                           # noticeable WiFi contention with other devices on 2026-09-27
SESSION_GAP = 15 * 60      # seconds of "nobody" before a Beeminder-attribution session is over
FACE_TOLERANCE = 0.6       # face_recognition's default; lower = stricter
FACE_LOCATE_MAX_DIM = 640  # downscale target for the cheap first-pass face *location* search
PHOTO_RETENTION_DAYS = 7   # how long to keep archived presence photos
PRESENCE_RETENTION_HOURS = 48  # how long to keep the per-second presence log (~3MB/day)
PRESENT_WINDOW = 5         # seconds after a camera last saw someone that a tick still counts
                           # them as present -- ticks run every ~1-3s, so this keeps a one-second
                           # sighting between two ticks from being missed
BEEMINDER_RETRY = 60       # seconds between attempts to log a matched session whose Beeminder
                           # datapoint couldn't be created yet (network down, API error)
PRESENCE_STALE = 30        # seconds without a presence update before a tick warns that the
                           # presence timer seems to have stopped
PLUG_TIMEOUT = 10 * 60     # seconds; refreshed on every poll while present, so the Shelly
                           # itself turns the plugs off if this process dies or hangs

FACES_DIR = Path(__file__).parent / "local" / "faces" / "joe"
GYM_GOAL = "gymvisit"
BEEMINDER_API = "https://www.beeminder.com/api/v1"
# NTFY_TOPIC comes from the environment (~/.local_bashrc), not hardcoded here -- ntfy's free
# tier has no real access control, an unguessable topic name IS the whole security model, so
# it can't sit in a repo that might ever be public.

# XDG state dir: logs and the rolling photo archive are runtime state, not
# config and not something to back up, so they don't belong in the repo.
STATE_DIR = Path.home() / ".local" / "state" / "gym-camera"
LOG_DIR = STATE_DIR / "logs"
PHOTOS_DIR = STATE_DIR / "photos"
STATE_FILE = STATE_DIR / "state.json"
PRESENCE_FILE = STATE_DIR / "presence.json"
PRESENCE_LOG_DIR = STATE_DIR / "presence"
FACE_CACHE = STATE_DIR / "known_faces.npz"

log = logging.getLogger("gym_loop")

# What a failed HTTP call to ntfy or Beeminder can raise. URLError alone misses read timeouts
# (TimeoutError), dropped connections (http.client.HTTPException, ConnectionError) and a
# garbled body (ValueError from json) -- any of which would otherwise crash the tick.
NET_ERRORS = (OSError, ValueError, http.client.HTTPException)


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


def archive_photo(jpeg_bytes: bytes, camera_key: str) -> None:
    """Save a presence-time photo and prune anything older than the retention window."""
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    (PHOTOS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{camera_key}.jpg").write_bytes(jpeg_bytes)
    cutoff = time.time() - PHOTO_RETENTION_DAYS * 86400
    for path in PHOTOS_DIR.glob("*.jpg"):
        if path.stat().st_mtime < cutoff:
            path.unlink()


def notify(message: str) -> None:
    topic = ({**_bashrc_exports(), **os.environ}).get("NTFY_TOPIC")
    if not topic:
        log.warning("NTFY_TOPIC not set, can't notify")
        return
    try:
        # The Title marks this as automation: prod's bin/ntfy-claude answers any untitled message
        # on the topic as if Joe typed it on his phone, so every automatic message needs a Title.
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}",
                                                        data=message.encode(),
                                                        headers={"Title": "Gym camera"}), timeout=10)
    except NET_ERRORS as e:
        log.warning(f"notify failed: {e}")


def load_known_faces(faces_dir: Path) -> list[np.ndarray]:
    """Encodings of the reference photos, from FACE_CACHE when the photos haven't changed
    since it was written. Every cycle is a fresh process, so without the cache each face check
    would re-encode every reference photo (~3.5 min for 27 of them, mostly the CNN fallback)."""
    import face_recognition
    import numpy as np

    paths = sorted(faces_dir.glob("*.jpg")) + sorted(faces_dir.glob("*.png"))
    signature = json.dumps([[p.name, p.stat().st_mtime, p.stat().st_size] for p in paths])
    try:
        cached = np.load(FACE_CACHE)
        if str(cached["signature"]) == signature:
            return list(cached["encodings"])
    except (OSError, KeyError, ValueError):
        pass

    from PIL import Image

    encodings = []
    for path in paths:
        image = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(image)
        if not locations:
            # HOG misses faces seen from steeply above, which is most of what the Between Doors
            # camera sees. dlib's CNN finds them but costs ~11s per 640px frame on this CPU --
            # too slow for live checks, fine for a cache rebuild.
            h, w = image.shape[:2]
            scale = min(1.0, FACE_LOCATE_MAX_DIM / max(h, w))
            small = np.array(Image.fromarray(image).resize((int(w * scale), int(h * scale))))
            locations = [tuple(int(v / scale) for v in loc)
                         for loc in face_recognition.face_locations(small, model="cnn")]
        if locations:
            encodings.append(face_recognition.face_encodings(image, known_face_locations=locations[:1])[0])
        else:
            log.warning(f"no face found in {path}, skipping")
    log.info(f"encoded {len(encodings)} reference face(s) from {faces_dir}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FACE_CACHE.with_suffix(".tmp.npz")
    np.savez(tmp, signature=np.array(signature),
             encodings=np.array(encodings).reshape(len(encodings), 128))
    tmp.replace(FACE_CACHE)  # live cycles read this; never let one see half a file
    return encodings


def matches_joe(jpeg_bytes: bytes, known: list[np.ndarray]) -> bool:
    """Two-stage: locate faces on a cheap downscaled copy (~0.2s regardless of the source
    resolution), then encode only those regions from the original full-res image (~0.15s per
    face). This is what makes ~1Hz checking affordable -- running detection on the full
    2560x1920 frame directly costs ~3s on its own, every time.

    Logs every attempt, not just successes -- a check that ran and found no usable face is a
    different, useful fact from one that found and matched a face, and both are different from
    this process not running at all."""
    import face_recognition
    import numpy as np
    from PIL import Image

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


class LookupFailed(Exception):
    pass


def find_todays_datapoint(auth: tuple[str, str]) -> dict | None:
    """A second gym session later the same day should extend today's existing datapoint, not
    create a sibling -- one entry per day, not one per session. The goal sums datapoints per
    day, so a sibling would count the day twice. That's why a failed lookup raises
    LookupFailed instead of returning None: None means "safe to create one".
    Compares against the local date, which matches Beeminder's daystamp only because the
    goal's deadline is midnight in this machine's timezone (checked 2026-09-30)."""
    token, user = auth
    today = time.strftime("%Y%m%d")
    url = f"{BEEMINDER_API}/users/{user}/goals/{GYM_GOAL}/datapoints.json?{urllib.parse.urlencode({'auth_token': token, 'count': 10})}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            points = json.loads(r.read())
    except NET_ERRORS as e:
        raise LookupFailed(f"failed to check for today's {GYM_GOAL} datapoint: {e}") from e
    return next((p for p in points if p.get("daystamp") == today), None)


def log_gymvisit(comment: str, dry_run: bool) -> tuple[str | None, str]:
    """Creates today's datapoint, or finds it if an earlier session today already made one.
    Returns (datapoint_id, existing_comment) -- existing_comment is '' for a fresh datapoint,
    so update_gymvisit_duration knows whether to set the comment or extend it. A None
    datapoint_id means it didn't get logged; the tick retries every BEEMINDER_RETRY."""
    if dry_run:
        log.info(f"[dry-run] would log {GYM_GOAL}: {comment}")
        return None, ""
    auth = _beeminder_auth()
    if not auth:
        return None, ""
    try:
        existing = find_todays_datapoint(auth)
    except LookupFailed as e:
        log.error(str(e))
        return None, ""
    if existing:
        log.info(f"today's {GYM_GOAL} datapoint already exists (id {existing['id']}), extending it")
        return existing["id"], existing.get("comment", "")
    token, user = auth
    url = f"{BEEMINDER_API}/users/{user}/goals/{GYM_GOAL}/datapoints.json"
    form = urllib.parse.urlencode({"auth_token": token, "value": 1, "comment": comment}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, form), timeout=30) as r:
            point = json.loads(r.read())
        log.info(f"logged {GYM_GOAL}: {comment} (id {point['id']})")
        return point["id"], ""
    except NET_ERRORS as e:
        log.error(f"failed to log {GYM_GOAL}: {e}")
        return None, ""


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
    except NET_ERRORS as e:
        log.error(f"failed to update {GYM_GOAL} datapoint {datapoint_id}: {e}")


def refresh_plugs() -> None:
    """Turn the GYM_PLUGS on and (re)arm their own PLUG_TIMEOUT auto-off, on the Shelly itself.
    Called on every cycle where a person is present, regardless of whose face (if anyone's)
    that turns out to be — this is the presence mode, not the identity mode. Because the
    timer lives on the plug's own firmware, the plugs still turn themselves off
    within PLUG_TIMEOUT even if cycles stop running or the network drops. Plugs are refreshed
    in parallel with a short timeout, so one unreachable plug neither stops the other nor
    makes the presence cycle overrun its second."""
    def refresh(key: str, host: str) -> None:
        try:
            Plug(key, host).set(True, toggle_after=PLUG_TIMEOUT, timeout=PRESENCE_TIMEOUT)
        except Exception as e:
            log.warning(f"plug {key} refresh failed: {e}")
    with ThreadPoolExecutor(len(GYM_PLUGS)) as pool:
        list(pool.map(refresh, GYM_PLUGS, GYM_PLUGS.values()))


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
    existing_comment: str = ""  # non-empty if today's datapoint predates this session
    last_beeminder_attempt: float = 0.0

    def start(self, now: float) -> None:
        self.active, self.matched, self.datapoint_id = True, False, None
        self.existing_comment = ""
        self.started_at = now
        self.last_face_check = 0.0
        log.info("session start")

    def log_match(self, now: float, dry_run: bool) -> None:
        """Create (or find) today's datapoint for this matched session."""
        self.last_beeminder_attempt = now
        self.datapoint_id, self.existing_comment = log_gymvisit(
            "gym camera auto-detect (session in progress)", dry_run)

    def needs_beeminder_retry(self, now: float, dry_run: bool) -> bool:
        """Matched, but the datapoint never got created -- without a retry, a network blip at
        the moment of the match would silently lose the whole visit."""
        return (self.matched and not self.datapoint_id and not dry_run
                and now - self.last_beeminder_attempt >= BEEMINDER_RETRY)

    def end(self, now: float, dry_run: bool) -> None:
        # last_seen, not now: duration is first-seen to last-seen, not including the
        # trailing SESSION_GAP wait before we're confident you've actually left.
        minutes = (self.last_seen - self.started_at) / 60
        log.info(f"session end ({minutes:.0f} min)")
        if self.matched and not self.datapoint_id and not dry_run:
            self.log_match(now, dry_run)  # one last try
            if not self.datapoint_id:
                log.error(f"session matched but never got logged to {GYM_GOAL}; log it by hand")
        if self.matched:
            comment = (f"{self.existing_comment}; +{minutes:.0f} min" if self.existing_comment
                       else f"gym camera auto-detect, ~{minutes:.0f} min")
            update_gymvisit_duration(self.datapoint_id, comment, dry_run)
        self.active = False


def _write_json_atomic(path: Path, data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)  # atomic, so a crash mid-write (or a reader mid-write) never sees half a file


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except ValueError as e:
        # A corrupt state file costs at most some bookkeeping; refusing to run over it would
        # cost the plugs too.
        log.warning(f"ignoring unreadable {path}: {e}")
        return {}


def log_presence_line(now: float, answers: dict[str, str]) -> None:
    """Append one line to this hour's presence log, and prune files past the retention window."""
    PRESENCE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = PRESENCE_LOG_DIR / f"{time.strftime('%Y%m%d-%H', time.localtime(now))}.tsv"
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write("time\t" + "\t".join(CAMERAS) + "\n")
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"
        f.write(stamp + "\t" + "\t".join(answers[k] for k in CAMERAS) + "\n")
    if new:  # once an hour is plenty
        cutoff = now - PRESENCE_RETENTION_HOURS * 3600
        for old in PRESENCE_LOG_DIR.glob("*.tsv"):
            if old.stat().st_mtime < cutoff:
                old.unlink()


def presence() -> None:
    """One presence cycle. Polls every camera, logs each one's answer, refreshes the plugs if
    anyone's seen, and leaves the result in PRESENCE_FILE for ticks. An unreachable camera
    counts as not seeing anyone; going offline and coming back are each logged once."""
    setup_logging()
    prev = _read_json(PRESENCE_FILE)
    now = time.time()

    def ask(host: str) -> str | Exception:
        try:
            return "person" if Camera.from_env(host).sees_person(timeout=PRESENCE_TIMEOUT) else "nobody"
        except Exception as e:
            return e
    # In parallel, so one camera timing out doesn't push the other past this second.
    with ThreadPoolExecutor(len(CAMERAS)) as pool:
        results = dict(zip(CAMERAS, pool.map(ask, CAMERAS.values())))
    answers = {}
    for key, result in results.items():
        was_offline = key in prev.get("offline", [])
        if isinstance(result, Exception):
            answers[key] = "offline"
            if not was_offline:
                log.warning(f"camera {key} offline: {result}")
        else:
            answers[key] = result
            if was_offline:
                log.info(f"camera {key} back online")
    seen_by = [k for k, v in answers.items() if v == "person"]
    if seen_by:
        refresh_plugs()  # presence mode: on/refresh regardless of identity; first, so a
                         # failure writing the logs below can't cost the plugs
    log_presence_line(now, answers)

    if seen_by != prev.get("seen_by", []):
        # Independent of matches_joe()'s own logging, which stops firing once matched --
        # without this, a long stretch of the cameras simply not seeing anyone is
        # indistinguishable in the log from a long stretch of normal silence.
        log.info(f"presence: seen by {', '.join(seen_by)}" if seen_by else "presence: no longer detected")

    _write_json_atomic(PRESENCE_FILE, {
        "at": now,
        "seen_by": seen_by,
        "offline": [k for k, v in answers.items() if v == "offline"],
        "last_seen": now if seen_by else prev.get("last_seen", 0.0),
        # start of the current unbroken run of sightings; ticks use it as a session's start
        "since": (prev.get("since", now) if prev.get("seen_by") else now) if seen_by else 0.0,
    })


@dataclass
class State:
    """Everything one tick hands to the next, via STATE_FILE."""
    session: Session = field(default_factory=Session)
    face_turn: int = 0  # index into CAMERAS for the next snapshot; advances on every attempt
    presence_stale: bool = False  # whether the last tick already warned about stale presence

    @classmethod
    def load(cls) -> State:
        raw = _read_json(STATE_FILE)
        try:
            known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__ and k != "session"}
            return cls(session=Session(**raw.get("session", {})), **known)
        except TypeError as e:
            log.warning(f"ignoring unreadable {STATE_FILE}: {e}")
            return cls()

    def save(self) -> None:
        _write_json_atomic(STATE_FILE, asdict(self))


def face_check(state: State, now: float, dry_run: bool) -> None:
    """Snap from the next camera in rotation, archive it, and if still unmatched, check it.
    Alternating cameras rather than snapping both keeps the WiFi cost at one ~400KB pull per
    check, and the rotation doesn't skip cameras whose person detector sees nobody, since face
    detection may catch a face their person detector didn't. A camera that can't be reached
    costs its turn."""
    session = state.session
    session.last_face_check = now
    keys = list(CAMERAS)
    key = keys[state.face_turn % len(keys)]
    state.face_turn += 1
    try:
        photo = Camera.from_env(CAMERAS[key]).snap()
    except Exception as e:
        log.warning(f"snap from {key} failed: {e}")
        return
    archive_photo(photo, key)
    log.info(f"snapped from {key}")
    if not session.matched and matches_joe(photo, load_known_faces(FACES_DIR)):
        session.matched = True
        notify(f"Gym camera ({key}) recognized you — this session will count.")
        session.log_match(now, dry_run)


def tick(dry_run: bool) -> None:
    setup_logging()
    state = State.load()
    session = state.session
    now = time.time()

    p = _read_json(PRESENCE_FILE)
    stale = now - p.get("at", 0.0) > PRESENCE_STALE
    if stale != state.presence_stale:
        if stale:
            log.warning(f"no presence update for over {PRESENCE_STALE}s -- is gym-presence.timer running?")
        else:
            log.info("presence updates resumed")
        state.presence_stale = stale
    last_seen = p.get("last_seen", 0.0)
    present = now - last_seen <= PRESENT_WINDOW

    if present:
        if not session.active:
            session.start(p.get("since") or last_seen)
        session.last_seen = max(session.last_seen, last_seen)
        # Fast while still trying to identify who this is; slow once that's settled --
        # each snap is a real ~400KB pull over the camera's WiFi link, not free bandwidth.
        interval = FACE_CHECK_INTERVAL if not session.matched else ARCHIVE_INTERVAL
        if now - session.last_face_check >= interval:
            face_check(state, now, dry_run)
    elif session.active and now - session.last_seen >= SESSION_GAP:
        session.end(now, dry_run)
    if session.active and session.needs_beeminder_retry(now, dry_run):
        log.info(f"retrying {GYM_GOAL} log for this matched session")
        session.log_match(now, dry_run)

    state.save()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("presence", help="run one presence/plugs cycle, then exit")
    t = sub.add_parser("tick", help="run one session/face/Beeminder cycle, then exit")
    t.add_argument("--dry-run", action="store_true", help="print what would be logged instead of calling Beeminder")
    a = p.parse_args()
    try:
        presence() if a.cmd == "presence" else tick(a.dry_run)
    except Exception:
        # The services drop info-level stderr (LogLevelMax=warning keeps one start per second
        # out of the journal), so a crash has to land in the log file to be seen at all.
        # systemd starts the next cycle either way.
        log.exception(f"{a.cmd} crashed")
        raise


if __name__ == "__main__":
    main()

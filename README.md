# gym-camera

Presence detection for a home gym: a Reolink camera's own onboard AI reports
whether anyone's there, and a periodic face-recognition check identifies
which known person it is. On a match, the gym fan and scent diffuser plugs turn on,
a phone notification confirms the match, and a Beeminder datapoint gets
logged for that person's gym-visit goal.

Currently runs on `zahlbericht` (the desktop); the plan is to move it to a
Raspberry Pi once the design is proven out.

## Files

- `gym.py` — low-level control: snapshots and AI state from both cameras, Shelly plug
  on/off/status. Stdlib only.
- `gym_loop.py` — the presence -> plugs -> Beeminder loop. Needs the `gym`
  conda env (`onnxruntime`, `opencv-python-headless`, `face_recognition`).
- `local/` — gitignored. `local/faces/<person>/*.jpg` holds each known
  person's reference photos. Never committed: these are photos of real
  people (currently just Joe; Tea and friends may be added later), and this
  repo may end up public like `prod` is.

## Credentials

Reads `REOLINK_USER`, `REOLINK_PASSWORD`, `BEEMINDER_AUTH_TOKEN`,
`BEEMINDER_USER`, `HILBERT_PUSH_KEY` from `~/.local_bashrc`, same convention as
`prod`. `HILBERT_PUSH_KEY` is the key for the house's phone notifications on
noether (`http://noether:8203`, hilbert-sdk's `automations/services/phone_push.py`),
which put the "recognized you" notification on Joe's phone.

## Running

    conda run -n gym python3 gym_loop.py presence         # one presence cycle (refreshes plugs if someone's there)
    conda run -n gym python3 gym_loop.py tick --dry-run   # one cycle, without logging to Beeminder
    conda run -n gym python3 gym_loop.py tick             # one cycle, for real

`gym_loop.py` never loops by itself. Each run does one cycle and exits, and
two systemd user units (in `~/.config/systemd/user/`, not this repo) supply
the loop:

- `gym-presence.timer` runs `gym_loop.py presence` every second. It polls
  both cameras, refreshes the plugs if anyone's seen, and appends a line to
  `~/.local/state/gym-camera/presence/` (kept 48 hours).
- `gym-camera.service` runs `gym_loop.py tick` with `Restart=always`. It reads
  the latest presence, tracks the session, runs face checks and logs to
  Beeminder.

A crash in either just means the next cycle runs from the last saved state.

# gym-camera

Presence detection for a home gym: a Reolink camera's own onboard AI reports
whether anyone's there, and a periodic face-recognition check identifies
which known person it is. On a match, plugs turn on (fan, scent diffuser),
a phone notification confirms the match, and a Beeminder datapoint gets
logged for that person's gym-visit goal.

Currently runs on `zahlbericht` (the desktop); the plan is to move it to a
Raspberry Pi once the design is proven out.

## Files

- `gym.py` — low-level control: camera snapshots and AI state, Shelly plug
  on/off/status. Stdlib only.
- `gym_loop.py` — the presence -> plugs -> Beeminder loop. Needs the `gym`
  conda env (`onnxruntime`, `opencv-python-headless`, `face_recognition`).
- `local/` — gitignored. `local/faces/<person>/*.jpg` holds each known
  person's reference photos. Never committed: these are photos of real
  people (currently just Joe; Tea and friends may be added later), and this
  repo may end up public like `prod` is.

## Credentials

Reads `REOLINK_USER`, `REOLINK_PASSWORD`, `BEEMINDER_AUTH_TOKEN`,
`BEEMINDER_USER`, `NTFY_TOPIC` from `~/.local_bashrc`, same convention as
`prod`. `NTFY_TOPIC` counts as a credential even though it doesn't look like
one: ntfy's free tier has no real access control, so the topic name being
unguessable is the only thing stopping someone else from posting
notifications to it.

## Running

    conda run -n gym python3 gym_loop.py run --dry-run   # watch without logging to Beeminder
    conda run -n gym python3 gym_loop.py run              # for real

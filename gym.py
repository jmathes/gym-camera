#!/usr/bin/env python3
"""gym — local control of the gym camera and smart plugs on zahlbericht.

    gym.py plug                      show both Shelly plugs and their on/off state and watts
    gym.py plug on|off NAME          switch one plug ("fan" / "aromadd", or its full name)
    gym.py snap OUT.jpg              save one 2560x1920 still from the Reolink camera
    gym.py person                    ask the camera's built-in AI if it sees a person now

Reolink needs REOLINK_USER and REOLINK_PASSWORD, from the environment or ~/.local_bashrc.
The Shelly plugs have no auth and speak plain HTTP RPC (Gen2+ API).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass

CAMERA_HOST = "192.168.86.39"  # gym.lan, Reolink RLC-510WA (MAC ec:71:db:ea:6c:f2)
PLUGS = {                      # Shelly Plug US Gen4 (S4PL-00116US)
    "fan": "192.168.86.37",      # "oscillating fan power", MAC e8:f6:0a:7d:ff:c8, ~58 W when on
    "aromadd": "192.168.86.36",  # "Aromadd power", MAC e8:f6:0a:7d:67:f4
}


def _get_json(url: str, data: bytes | None = None, timeout: float = 10) -> object:
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


@dataclass(frozen=True)
class Plug:
    key: str
    host: str

    def _rpc(self, method: str, **params) -> dict:
        q = urllib.parse.urlencode({k: json.dumps(v) if isinstance(v, (bool, dict)) else v
                                    for k, v in params.items()})
        return _get_json(f"http://{self.host}/rpc/{method}?{q}", timeout=5)

    def status(self) -> dict:
        return self._rpc("Switch.GetStatus", id=0)

    def name(self) -> str:
        return self._rpc("Switch.GetConfig", id=0)["name"] or self.key

    def set(self, on: bool) -> None:
        self._rpc("Switch.Set", id=0, on=on)


def find_plug(name: str) -> Plug:
    for key, host in PLUGS.items():
        p = Plug(key, host)
        if name.lower() in (key, p.name().lower()):
            return p
    sys.exit(f"no plug named {name!r}; have {list(PLUGS)}")


def _bashrc_exports(path: str = "~/.local_bashrc") -> dict[str, str]:
    """`export NAME=value` lines from the tokens file, so cron jobs see them too."""
    out = {}
    try:
        with open(os.path.expanduser(path)) as f:
            for line in f:
                m = re.match(r"\s*export\s+([A-Z_][A-Z0-9_]*)=(.*)", line)
                if m:
                    out[m[1]] = m[2].strip().strip("'\"")
    except FileNotFoundError:
        pass
    return out


@dataclass(frozen=True)
class Camera:
    host: str
    user: str
    password: str

    @classmethod
    def from_env(cls, host: str) -> Camera:
        env = {**_bashrc_exports(), **os.environ}
        for var in ("REOLINK_USER", "REOLINK_PASSWORD"):
            if var not in env:
                sys.exit(f"{var} not set")
        return cls(host, env["REOLINK_USER"], env["REOLINK_PASSWORD"])

    def _url(self, **q) -> str:
        q |= {"user": self.user, "password": self.password}
        return f"http://{self.host}/cgi-bin/api.cgi?{urllib.parse.urlencode(q)}"

    def api(self, cmd: str, param: dict | None = None) -> dict:
        body = json.dumps([{"cmd": cmd, "action": 0, "param": param or {}}]).encode()
        res = _get_json(self._url(cmd=cmd), body)[0]
        if res["code"] != 0:
            raise RuntimeError(f"{cmd}: {res.get('error')}")
        return res["value"]

    def snap(self) -> bytes:
        """One JPEG via the CGI Snap command."""
        with urllib.request.urlopen(self._url(cmd="Snap", channel=0, rs="gym"), timeout=15) as r:
            body = r.read()
        if body[:2] != b"\xff\xd8":
            sys.exit(f"camera did not return a JPEG: {body[:300]!r}")
        return body

    def sees_person(self) -> bool:
        """The RLC-510WA runs its own person/vehicle/pet detector; this reads its current verdict."""
        return bool(self.api("GetAiState", {"channel": 0})["people"]["alarm_state"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plug", help="show or switch the Shelly plugs")
    pp.add_argument("action", nargs="?", choices=["on", "off"])
    pp.add_argument("name", nargs="?")
    ps = sub.add_parser("snap", help="save one camera still")
    ps.add_argument("out")
    sub.add_parser("person", help="does the camera see a person right now?")
    a = p.parse_args()

    if a.cmd == "plug":
        if a.action:
            if not a.name:
                sys.exit("say which plug")
            find_plug(a.name).set(a.action == "on")
        for key, host in PLUGS.items():
            plug = Plug(key, host)
            s = plug.status()
            print(f"{'ON ' if s['output'] else 'off'}  {s['apower']:6.1f} W  {plug.name()}  ({host})")
    elif a.cmd == "snap":
        with open(a.out, "wb") as f:
            f.write(Camera.from_env(CAMERA_HOST).snap())
        print(f"wrote {a.out}")
    else:
        print("person" if Camera.from_env(CAMERA_HOST).sees_person() else "nobody")


if __name__ == "__main__":
    main()

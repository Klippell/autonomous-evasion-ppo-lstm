"""Webots path setup and static world metadata for the evader environment."""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass


WEBOTS_HOME = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")


@dataclass(frozen=True)
class SpawnPose:
    evader_xy: tuple[float, float]
    pursuer_xy: tuple[float, float]
    heading: float


DEFAULT_SPAWN_POSES = (
    SpawnPose((-45.0, 46.28), (-22.98, 45.88), math.pi),
    SpawnPose((45.0, -45.0), (25.0, -45.0), 0.0),
    SpawnPose((-105.0, 4.5), (-85.0, 4.5), math.pi),
    SpawnPose((105.0, 93.0), (85.0, 93.0), 0.0),
)


def configure_webots_paths() -> None:
    os.environ.setdefault("WEBOTS_HOME", WEBOTS_HOME)
    for path in (
        os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin"),
        os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin", "cpp"),
        os.path.join(WEBOTS_HOME, "lib", "controller"),
        os.path.join(WEBOTS_HOME, "projects", "vehicles", "lib"),
    ):
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except (AttributeError, FileNotFoundError, OSError):
                pass
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

    controller_python = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
    if os.path.exists(controller_python) and controller_python not in sys.path:
        sys.path.append(controller_python)

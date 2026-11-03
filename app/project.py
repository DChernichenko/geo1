# -*- coding: utf-8 -*-
"""Сериализация проекта (JSON).

В проекте хранятся:
  * путь к файлу изображения;
  * выбранная система координат и её параметры (зона, масштаб, EPSG);
  * пиксельные координаты опорных точек и их "мировые" (трёхмерные) координаты;
  * отмеченные пользователем пиксели/зоны и оценки их координат
    с доверительными интервалами.
"""

import json
import os

PROJECT_EXT = ".cooproj"
PROJECT_VERSION = 1


def _clean(v):
    """numpy -> python, NaN -> None (json не принимает NaN корректно)."""
    import math
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def save_project(path, image_path, sys_key, zone, scale, custom_epsg,
                 control_points, marked_points):
    data = {
        "version": PROJECT_VERSION,
        "image": os.path.abspath(image_path) if image_path else "",
        "system": {"key": sys_key, "zone": zone, "scale": scale,
                   "custom_epsg": custom_epsg},
        "control_points": [
            {"u": _clean(p["u"]), "v": _clean(p["v"]),
             "c1": _clean(p["c1"]), "c2": _clean(p["c2"]),
             "h": _clean(p["h"]), "sigma": _clean(p.get("sigma", 0.5)),
             "valid": bool(p.get("valid", True))}
            for p in control_points],
        "marked_points": [
            {"u": _clean(m["u"]), "v": _clean(m["v"]),
             "kind": m.get("kind", "pixel"),
             "rect": ([_clean(x) for x in m["rect"]] if m.get("rect") else None),
             "corner_strength": _clean(m.get("corner_strength")),
             "result": ({
                 "world": [_clean(x) for x in m["result"]["world"]],
                 "latlonh": ([_clean(x) for x in m["result"]["latlonh"]]
                             if m["result"].get("latlonh") else None),
                 "ci_world": ([_clean(x) for x in m["result"]["ci_world"]]
                              if m["result"].get("ci_world") else None),
                 "ci_latlon_m": ([_clean(x) for x in m["result"]["ci_latlon_m"]]
                                 if m["result"].get("ci_latlon_m") else None)}
                 if m.get("result") else None)}
            for m in marked_points],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_project(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "control_points" not in data:
        raise ValueError("Файл не является проектом программы.")
    sysd = data.get("system", {})
    cps = []
    for p in data.get("control_points", []):
        cps.append({"u": float(p["u"]), "v": float(p["v"]),
                    "c1": float(p["c1"]), "c2": float(p["c2"]),
                    "h": float(p.get("h", 0.0)),
                    "sigma": float(p.get("sigma", 0.5) or 0.5),
                    "valid": bool(p.get("valid", True))})
    mps = []
    for m in data.get("marked_points", []):
        mps.append({"u": float(m["u"]), "v": float(m["v"]),
                    "kind": m.get("kind", "pixel"),
                    "rect": ([float(x) for x in m["rect"]] if m.get("rect") else None),
                    "corner_strength": m.get("corner_strength"),
                    "result": m.get("result")})
    return {
        "image": data.get("image", ""),
        "sys_key": sysd.get("key", "wgs84_geo"),
        "zone": sysd.get("zone"),
        "scale": sysd.get("scale"),
        "custom_epsg": sysd.get("custom_epsg"),
        "control_points": cps,
        "marked_points": mps,
    }

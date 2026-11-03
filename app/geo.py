# -*- coding: utf-8 -*-
"""
Геодезическое ядро программы оценки координат по изображению с БПЛА.

Подход:
  * Опорные точки вводятся пользователем в выбранной системе координат
    (WGS-84, ПЗ-90.27, СК-42 — геодезические или гауссовы проекции, либо
    произвольная пользовательская СК) и переводятся в геоцентрическую
    систему WGS-84 (ECEF).
  * Вокруг первой опорной точки строится локальная топоцентрическая
    система ENU (Восток-Север-Вверх). Все точки проецируются в неё.
  * Связь "пиксель -> плоскость ENU" оценивается гомографией (DLT, МНК),
    высота — аффинным преобразованием. Невязки служат оценкой шума,
    ковариация коэффициентов вычисляется линеаризацией (дельта-метод),
    поэтому для любого пикселя можно получить доверительный интервал
    трёхмерных координат (95 %, k = 1.96).
  * Результат пересчитывается в целевую систему координат; переходы
    между датумами выполняются формулой Гельмерта (координатная
    конвенция), Гаусс–Крюгер реализован прямыми/обратными рядовыми
    формулами (не требует сетевых ресурсов PROJ).
"""

import math
import re

import numpy as np

try:                                        # pyproj используется только для
    from pyproj import CRS, Transformer     # произвольных EPSG-кодов
except Exception:                           # пользователя
    CRS = None
    Transformer = None

# ---------------------------------------------------------------------------
# Эллипсоиды
# ---------------------------------------------------------------------------
WGS84_A, WGS84_F = 6378137.0, 1.0 / 298.257223563
KRASS_A, KRASS_F = 6378245.0, 1.0 / 298.3
PZ90_A, PZ90_F = 6378136.0, 1.0 / 298.257222101

ELLIPSOIDS = {"WGS84": (WGS84_A, WGS84_F),
              "SK42": (KRASS_A, KRASS_F),
              "PZ90": (PZ90_A, PZ90_F)}

# 7 параметров Гельмерта (конвенция координатной системы), src -> WGS-84:
#   X_wgs = T + s * R * X_src
HELMERT_TO_WGS84 = {
    "SK42": (24.6802, 13.1412, 4.3460, 0.0, 0.0, 0.0, 0.0),
    "PZ90": (-20.4086, 0.0424, -2.4128, 0.0, 0.0, 0.0, 0.0),
    "WGS84": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}

CONF_K = 1.96          # коэффициент доверительного интервала 95 %, k = 1.96
SEC2RAD = math.pi / (180.0 * 3600.0)   # 1 секунда дуги в радианах


def _helmert_matrix(rx_deg, ry_deg, rz_deg, s_ppm):
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    s = 1.0 + s_ppm * 1e-6
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [s * cz,        -s * sz,        0.0],
        [s * sz * cy,   s * cz * cy,   -sy],
        [s * sz * sx - cz * cy * cx, -s * cz * sx - sz * cy * cx, s * cx],
    ])


def geodetic_to_ecef(lat_deg, lon_deg, h, a=WGS84_A, f=WGS84_F):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = f * (2.0 - f)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    return np.array([(N + h) * cos_lat * math.cos(lon),
                     (N + h) * cos_lat * math.sin(lon),
                     (N * (1.0 - e2) + h) * sin_lat])


def ecef_to_geodetic(xyz, a=WGS84_A, f=WGS84_F):
    """Обратная геодезическая задача (итерации по широте, точность ~1e-12 рад)."""
    e2 = f * (2.0 - f)
    b = a * (1.0 - f)
    ep2 = (a * a - b * b) / (b * b)
    x, y, z = (float(v) for v in xyz)
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    if p < 1e-9:                                  # на оси вращения
        return (math.copysign(90.0, z), 0.0, abs(z) - b)
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(z + ep2 * b * math.sin(theta) ** 3,
                     p - e2 * a * math.cos(theta) ** 3)
    for _ in range(8):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        new = math.atan2(z + e2 * N * sin_lat, p)
        if abs(new - lat) < 1e-13:
            lat = new
            break
        lat = new
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - N if abs(lat) < math.radians(80.0) \
        else abs(z) / sin_lat - N * (1.0 - e2)
    return math.degrees(lat), math.degrees(lon), h


def datum_to_wgs84_ecef(datum, xyz):
    key = str(datum).upper()
    if key == "WGS84":
        return np.asarray(xyz, dtype=float)
    tx, ty, tz, rx, ry, rz, s = HELMERT_TO_WGS84[key]
    R = _helmert_matrix(rx, ry, rz, s)
    return np.array([tx, ty, tz]) + R @ np.asarray(xyz, dtype=float)


def wgs84_ecef_to_datum(datum, xyz):
    key = str(datum).upper()
    if key == "WGS84":
        return np.asarray(xyz, dtype=float)
    tx, ty, tz, rx, ry, rz, s = HELMERT_TO_WGS84[key]
    R = _helmert_matrix(rx, ry, rz, s)
    return np.linalg.inv(R) @ (np.asarray(xyz, dtype=float) - np.array([tx, ty, tz]))


# ---------------------------------------------------------------------------
# Разбор/форматирование углов
# ---------------------------------------------------------------------------
_DMS_RE = re.compile(
    r"^(\d{1,3})\s*[°д]\s*(\d{1,2})\s*['′мин]*\s*(\d{1,2}(?:[.,]\d+)?)?\s*[\"″сек]*"
    r"\s*([NSEWnsew北ювz])?$")


def parse_angle(text):
    """'55.75', '55°45\'00"', '55°45' -> float градусов."""
    s = str(text).strip().replace(",", ".")
    m = _DMS_RE.match(s)
    if m:
        v = float(m.group(1)) + float(m.group(2)) / 60.0
        if m.group(3):
            v += float(m.group(3)) / 3600.0
        hemi = (m.group(4) or "").upper()
        if hemi in ("S", "W", "Ю", "З"):
            v = -v
        return v
    return float(s)


def format_dms(value, is_lat=True):
    sign = "-" if value < 0 else ""
    v = abs(value)
    d = int(v)
    mf = (v - d) * 60.0
    m = int(mf)
    sec = (mf - m) * 60.0
    hemi = ("N" if value >= 0 else "S") if is_lat else ("E" if value >= 0 else "W")
    return "%s%d°%02d′%05.2f″%s" % (sign, d, m, sec, hemi)


# ---------------------------------------------------------------------------
# Описания систем координат
# ---------------------------------------------------------------------------
class SysDef:
    def __init__(self, key, title, kind, datum="WGS84", units="deg"):
        self.key = key
        self.title = title
        self.kind = kind              # 'geodetic' | 'gauss' | 'custom'
        self.datum = datum.upper()
        self.units = units            # 'deg' | 'm'

    @property
    def ellps(self):
        return ELLIPSOIDS[self.datum]


SYSTEMS = [
    SysDef("wgs84_geo", "WGS-84 (геодезические: широта, долгота, высота)", "geodetic", "WGS84"),
    SysDef("pz90_geo", "ПЗ-90.27 (геодезические: широта, долгота, высота)", "geodetic", "PZ90"),
    SysDef("sk42_geo", "СК-42 (геодезические: широта, долгота, высота)", "geodetic", "SK42"),
    SysDef("sk42_gk", "СК-42 / Гаусс–Крюгер (X, Y зоны, H; зоны 1..60)", "gauss", "SK42", "m"),
    SysDef("msk", "МСК-подобная (Гаусс–Крюгер + масштабный множитель k)", "gauss", "SK42", "m"),
    SysDef("wgs84_utm", "WGS-84 / UTM (зоны 1..60)", "gauss", "WGS84", "m"),
    SysDef("custom", "Произвольная пользовательская СК", "custom", "WGS84", "m"),
]
SYS_INDEX = {s.key: s for s in SYSTEMS}


class CustomSystem(SysDef):
    """Произвольная СК: либо параллельный перенос геоцентрика WGS-84
    (X0,Y0,Z0), либо любая проектная система по её EPSG-коду."""

    def __init__(self, name="Пользовательская", offset=(0.0, 0.0, 0.0), epsg=None):
        super().__init__("custom", name, "custom", "WGS84", "m")
        self.offset = np.array(offset, dtype=float)
        self.custom_epsg = epsg
        self._tr = None
        if epsg and Transformer is not None:
            try:
                self._tr = Transformer.from_crs(CRS.from_user_input(str(epsg)),
                                                CRS.from_user_input("EPSG:4979"),
                                                always_xy=True)
            except Exception:
                self._tr = None

    def to_ecef(self, c1, c2, h):
        if self._tr is not None:
            out = self._tr.transform([c1], [c2], [h])
            return np.array([float(out[0][0]), float(out[1][0]), float(out[2][0])])
        return np.array([c1, c2, h]) + self.offset

    def from_ecef(self, xyz):
        if self._tr is not None:
            out = self._tr.transform([xyz[0]], [xyz[1]], [xyz[2]])
            return float(out[0][0]), float(out[1][0]), float(out[2][0])
        v = np.asarray(xyz, dtype=float) - self.offset
        return float(v[0]), float(v[1]), float(v[2])


# ---------------------------------------------------------------------------
# Гаусс–Крюгер
# ---------------------------------------------------------------------------
def _merid_arc_coeffs(a, e2):
    e4, e6 = e2 * e2, e2 ** 3
    c0 = a * (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0)
    c2 = a * (-3.0 * e2 / 8.0 - 3.0 * e4 / 32.0 - 45.0 * e6 / 1024.0)
    c4 = a * (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0)
    c6 = a * (-35.0 * e6 / 3072.0)
    return c0, c2, c4, c6


def _merid_arc(B, a, e2):
    """Дуга меридиана от экватора до широты B (рад), метры."""
    c0, c2, c4, c6 = _merid_arc_coeffs(a, e2)
    return (c0 * B + c2 * math.sin(2 * B) + c4 * math.sin(4 * B)
            + c6 * math.sin(6 * B))


def _foot_latitude(xn, a, e2):
    """Широта основания (footpoint) из X (м) — итерациями (~1e-12 рад)."""
    c0, c2, c4, c6 = _merid_arc_coeffs(a, e2)
    B = xn / c0
    for _ in range(20):
        newB = (xn - c2 * math.sin(2 * B) - c4 * math.sin(4 * B)
                - c6 * math.sin(6 * B)) / c0
        if abs(newB - B) < 1e-15:
            B = newB
            break
        B = newB
    return B


def gauss_forward(lat_deg, lon_deg, lon0, a, f, k0=1.0):
    """lat,lon (в эллипсоиде датума) -> X(север), Y(восток, без номера зоны), м."""
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    dl = lon - math.radians(lon0)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    t = math.tan(lat)
    eta2 = ep2 * cos_lat * cos_lat
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    A = dl * cos_lat
    M = _merid_arc(lat, a, e2)
    x = k0 * (M + N * t / 2.0 * A ** 2
              + N * t * (5 - t * t + 9 * eta2 + 4 * eta2 ** 2) / 24.0 * A ** 4
              + N * t * (61 - 58 * t * t + t ** 4) / 720.0 * A ** 6)
    y = k0 * N * cos_lat * dl * (
        1.0 + (1 - t * t + eta2) / 6.0 * A ** 2
        + (5 - 18 * t * t + t ** 4 + 14 * eta2 - 58 * eta2 * t * t) / 120.0 * A ** 4)
    return x, y


def gauss_inverse(x, y, lon0, a, f, k0=1.0):
    """X, Y (м) -> lat, lon (градусы). Метод Ньютона 2x2 с ЧИСЛЕННЫМ
    якобианом рядов Гаусса–Крюгера и backtracking; квадратичная
    сходимость (~5 итераций), устойчив во всей 6-градусной зоне."""
    e2 = f * (2 - f)
    xn, yn = x / k0, y / k0
    l0r = math.radians(lon0)

    def xy(B, L):
        sin_l, cos_l = math.sin(B), math.cos(B)
        dl = L - l0r
        t = math.tan(B)
        t2 = t * t
        eta2 = (e2 / (1 - e2)) * cos_l * cos_l
        N = a / math.sqrt(1 - e2 * sin_l * sin_l)
        A = dl * cos_l
        M = _merid_arc(B, a, e2)
        xc = M + N * t * (A ** 2 / 2.0
                          + (5 - t2 + 9 * eta2 + 4 * eta2 ** 2) / 24.0 * A ** 4
                          + (61 - 58 * t2 + t2 * t2) / 720.0 * A ** 6)
        yc = N * cos_l * dl * (1.0 + (1 - t2 + eta2) / 6.0 * A ** 2
                               + (5 - 18 * t2 + t2 * t2 + 14 * eta2
                                  - 58 * eta2 * t2) / 120.0 * A ** 4)
        return xc, yc

    # старт: широта основания от X, долгота — из переноса по параллели
    xmax = _merid_arc(math.radians(89.9), a, e2)
    if xn >= 0:
        lat = _foot_latitude(min(xn, xmax), a, e2)
    else:
        lat = -_foot_latitude(min(-xn, xmax), a, e2)
    cos_start = max(abs(math.cos(lat)), 1e-3)
    lonr = l0r + yn / (a * cos_start)
    prev_res = None
    stall = 0
    for _ in range(30):
        xc, yc = xy(lat, lonr)
        fx, fy = xc - xn, yc - yn
        res = fx * fx + fy * fy
        if abs(fx) < 1e-7 and abs(fy) < 1e-7:
            break
        h = 1e-7
        xp, yp = xy(lat + h, lonr)
        xm, ym = xy(lat - h, lonr)
        xl, yl = xy(lat, lonr + h)
        xr, yr = xy(lat, lonr - h)
        J = np.array([[(xp - xm) / (2 * h), (xl - xr) / (2 * h)],
                      [(yp - ym) / (2 * h), (yl - yr) / (2 * h)]])
        try:
            d = np.linalg.solve(J, np.array([-fx, -fy]))
        except np.linalg.LinAlgError:
            break
        # backtracking: шаг должен уменьшать невязку
        step = 1.0
        ok = False
        nlat, nlon = lat, lonr
        for _t in range(10):
            clat = max(-math.radians(89.999), min(math.radians(89.999),
                                                  lat + step * d[0]))
            clon = lonr + step * d[1]
            cx, cy = xy(clat, clon)
            if (cx - xn) ** 2 + (cy - yn) ** 2 < res:
                nlat, nlon, ok = clat, clon, True
                break
            step *= 0.5
        if not ok:
            if prev_res is not None and res >= prev_res * (1.0 - 1e-12):
                stall += 1
            else:
                stall = 0
            if stall >= 2:
                break
        else:
            stall = 0
        prev_res = res
        lat, lonr = nlat, nlon
    return math.degrees(lat), math.degrees(lonr)


# ---------------------------------------------------------------------------
# Конвертер активной системы координат <-> ECEF(WGS-84)
# ---------------------------------------------------------------------------
class GeoConverter:
    def __init__(self, sys_key="wgs84_geo", zone=None, scale=None, custom=None):
        self.custom = None
        self.zone = None
        self.scale = None
        self.set_system(sys_key, zone=zone, scale=scale, custom=custom)

    def set_system(self, sys_key, zone=None, scale=None, custom=None):
        self.sys_key = sys_key
        if sys_key == "custom":
            self.sys = custom if custom is not None else CustomSystem()
        else:
            self.sys = SYS_INDEX.get(sys_key, SYSTEMS[0])
        self.zone = int(zone) if zone else None
        try:
            self.scale = float(scale) if scale not in (None, "", 0) else None
        except (TypeError, ValueError):
            self.scale = None

    # --- параметры зоны ------------------------------------------------
    @property
    def central_meridian(self):
        z = self.zone if self.zone else 31
        # единая конвенция ГГС/UTM: номер зоны N,CM = N*6 - 183
        return z * 6 - 183

    @property
    def k0(self):
        if self.sys.kind != "gauss":
            return 1.0
        if self.sys.key == "wgs84_utm":
            return 0.9996
        return self.scale if self.scale else 1.0

    @staticmethod
    def zone_for_lon(lon_deg, utm=False):
        n = int(math.floor((lon_deg + 180.0) / 6.0)) + 1
        if utm:
            n = ((n - 1) % 60) + 1
        return max(1, min(60, n))

    def strip_zone_prefix(self, y_value):
        """Срезает номер зоны (1..60) и ложный восток 5 000 000 м из Y вида
        375000000 (ГГС-традиция). Если префикса нет — возвращает Y как есть
        (ложный восток не добавлялся)."""
        yv = float(y_value)
        ay = abs(yv)
        if 10000000.0 <= ay < 610000000.0:      # Y с префиксом зоны (ГГС)
            n = int(ay // 10000000.0)
            if 1 <= n <= 60:
                return yv - (n * 10000000.0 + 5000000.0) * (1 if yv >= 0 else -1)
        return yv

    # --- пересчёты -------------------------------------------------------
    def world_to_ecef(self, c1, c2, h):
        kind = self.sys.kind
        if kind == "custom":
            return self.sys.to_ecef(c1, c2, h)
        a, f = self.sys.ellps
        if kind == "geodetic":
            xyz = geodetic_to_ecef(parse_angle(c1), parse_angle(c2),
                                   float(h), a, f)
        else:
            y = self.strip_zone_prefix(c2)
            lat, lon = gauss_inverse(float(c1), y, self.central_meridian,
                                     a, f, self.k0)
            # если зона не задана — определить по долготе и пересчитать
            # (стартовый CM мог быть неверным при >3° отклонения)
            if self.zone is None:
                self.ensure_zone(lon)
                lat, lon = gauss_inverse(float(c1), y, self.central_meridian,
                                         a, f, self.k0)
            xyz = geodetic_to_ecef(lat, lon, float(h), a, f)
        return datum_to_wgs84_ecef(self.sys.datum, xyz)

    def ecef_to_world(self, xyz):
        kind = self.sys.kind
        if kind == "custom":
            return self.sys.from_ecef(xyz)
        xw = wgs84_ecef_to_datum(self.sys.datum, np.asarray(xyz, dtype=float))
        a, f = self.sys.ellps
        lat, lon, h = ecef_to_geodetic(xw, a, f)
        if kind == "geodetic":
            return lat, lon, h
        x, y = gauss_forward(lat, lon, self.central_meridian, a, f, self.k0)
        z = self.zone if self.zone else self.zone_for_lon(lon, self.sys.key == "wgs84_utm")
        if self.sys.key == "wgs84_utm":
            return x, 500000.0 + y, h
        return x, z * 10000000.0 + 5000000.0 + y, h

    def ensure_zone(self, lon_deg):
        if self.sys.kind == "gauss" and not self.zone:
            self.zone = self.zone_for_lon(lon_deg, self.sys.key == "wgs84_utm")

    def format_world(self, c1, c2, h):
        if self.sys.kind == "geodetic":
            return "%s  %s  H=%.2f м" % (format_dms(c1, True), format_dms(c2, False), h)
        return "X=%.3f  Y=%.3f  H=%.2f м" % (c1, c2, h)

    def labels(self):
        if self.sys.kind == "geodetic":
            return ("Широта B°", "Долгота L°", "Высота H, м")
        if self.sys.kind == "custom":
            return ("C1 (м)", "C2 (м)", "H (м)")
        return ("X, км/м", "Y (с номером зоны), м", "H, м")


# ---------------------------------------------------------------------------
# Локальная топоцентрическая система ENU
# ---------------------------------------------------------------------------
def enu_basis(lat0_deg, lon0_deg):
    la, lo = math.radians(lat0_deg), math.radians(lon0_deg)
    sl, cl, sn, cn = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    E = np.array([-sn, cn, 0.0])
    N = np.array([-sl * cn, -sl * sn, cl])
    U = np.array([cl * cn, cl * sn, sl])
    return np.vstack([E, N, U])                 # строки — единичные векторы


def ecef_to_enu(origin, B, xyz):
    d = np.asarray(xyz, dtype=float) - origin
    return B @ d


def enu_to_ecef(origin, B, enu):
    return origin + B.T @ np.asarray(enu, dtype=float)


# ---------------------------------------------------------------------------
# Оценка связи "пиксель -> мир" (гомография + аффинная высота)
# ---------------------------------------------------------------------------
def _norm_dlt(pix, world):
    """Нормализованный DLT: возвращает (A3x3, pix_mean, pix_scale, world_mean, world_scale)."""
    pm = pix.mean(axis=0)
    ps = math.sqrt(max(float(np.mean((pix - pm) ** 2)), 1e-12))
    wm = world.mean(axis=0)
    ws = math.sqrt(max(float(np.mean((world - wm) ** 2)), 1e-12))
    pn = (pix - pm) / ps
    wn = (world - wm) / ws
    rows = []
    for (u, v), (x, y) in zip(pn, wn):
        w_ = 1.0
        rows.append([-u * w_, -v * w_, -w_, 0, 0, 0, x * u, x * v, x])
        rows.append([0, 0, 0, -u * w_, -v * w_, -w_, y * u, y * v, y])
    M = np.array(rows)
    _, _, Vt = np.linalg.svd(M)
    Ah = Vt[-1].reshape(3, 3)
    Ah = Ah / Ah[2, 2]
    return Ah, pm, ps, wm, ws


def _homography_jacobian(A, u, v):
    """dh/dA (2x9) и знаменатель w для точки (u,v)."""
    Xu, Yu = A[0, 0] * u + A[0, 1] * v + A[0, 2], A[1, 0] * u + A[1, 1] * v + A[1, 2]
    w = A[2, 0] * u + A[2, 1] * v + 1.0
    J = np.zeros((2, 9))
    du = np.array([u, v, 1.0])
    J[0, 0:3] = du / w
    J[1, 3:6] = du / w
    J[0, 6:9] = -Xu * du / (w * w)
    J[1, 6:9] = -Yu * du / (w * w)
    return J, w


def _affine_fit(pix, val, weights=None):
    A = np.hstack([pix, np.ones((len(pix), 1))])
    if weights is None:
        sol, res, rank, sv = np.linalg.lstsq(A, val, rcond=None)
    else:
        sw = np.sqrt(weights)[:, None]
        sol, res, rank, sv = np.linalg.lstsq(A * sw, val * np.sqrt(weights), rcond=None)
    return sol, A


class HomographyEstimator:
    """Строит модель pixel->ENU и распространяет ошибки на произвольный пиксель."""

    MIN_POINTS = 3

    def __init__(self, sigma_pix=0.5, default_sigma_obj=0.5):
        self.sigma_pix = sigma_pix
        self.default_sigma_obj = default_sigma_obj

    def fit(self, points):
        """points: [{'u','v','ecef':np.array,'sigma_obj'}, ...] (>=3).

        Возвращает словарь модели или None. Оценивается гомография
        пиксель -> (east, north) в ENU и аффинное преобразование для up.
        Коэффициенты уточняются взвешенным МНК с итерациями по невязкам
        (robust re-weighting), ковариация — линеаризацией.
        """
        pts = [p for p in points if p.get("valid", True)]
        if len(pts) < self.MIN_POINTS:
            return None
        ecef = np.array([p["ecef"] for p in pts], dtype=float)
        sig = np.array([float(p.get("sigma_obj", self.default_sigma_obj))
                        for p in pts], dtype=float)
        sig = np.clip(np.nan_to_num(sig, nan=self.default_sigma_obj), 1e-3, 1e6)
        origin = ecef[0]
        lat0, lon0, _ = ecef_to_geodetic(origin)
        B = enu_basis(lat0, lon0)
        enu = np.array([B @ (e - origin) for e in ecef])
        pix = np.array([[p["u"], p["v"]] for p in pts], dtype=float)
        n = len(pix)
        dof = max(n - 3, 1)

        # ---- начальная оценка: нормализованный DLT ----------------------
        Ah, pm, ps, wm, ws = _norm_dlt(pix, enu[:, :2])

        def denorm_predict(Ah_, u, v):
            un = (u - pm[0]) / ps
            vn = (v - pm[1]) / ps
            num = Ah_ @ np.array([un, vn, 1.0])
            return num[0] / num[2] * ws + wm[0], num[1] / num[2] * ws + wm[1]

        # ---- итеративный взвешенный МНК по невязкам (Gauss–Newton) ------
        weights = 1.0 / sig ** 2
        for _ in range(8):
            pred = np.array([denorm_predict(Ah, u, v) for u, v in pix])
            r = pred - enu[:, :2]
            rms = math.sqrt(float(np.mean(weights * np.sum(r ** 2, axis=1)) /
                                  float(np.mean(weights))))
            # robust-веса (Huber, порог 2.5σ)
            d = np.linalg.norm(r, axis=1)
            thr = max(2.5 * max(rms, 1e-9), 1e-9)
            rw = np.minimum(1.0, thr / np.maximum(d, 1e-12))
            w_i = weights * rw
            # локальная нормализация и переоценка DLT с весами
            rows = []
            for (u, v), (x, y), wi in zip(pix, enu[:, :2], np.sqrt(w_i)):
                un, vn = (u - pm[0]) / ps, (v - pm[1]) / ps
                xn, yn = (x - wm[0]) / ws, (y - wm[1]) / ws
                rows.append(wi * np.array([-un, -vn, -1.0, 0, 0, 0,
                                           xn * un, xn * vn, xn]))
                rows.append(wi * np.array([0, 0, 0, -un, -vn, -1.0,
                                           yn * un, yn * vn, yn]))
            M = np.array(rows)
            _, _, Vt = np.linalg.svd(M)
            Ah_new = Vt[-1].reshape(3, 3)
            Ah_new = Ah_new / Ah_new[2, 2]
            if np.allclose(Ah_new, Ah, rtol=1e-10, atol=1e-12):
                Ah = Ah_new
                break
            Ah = Ah_new

        pred = np.array([denorm_predict(Ah, u, v) for u, v in pix])
        resid = pred - enu[:, :2]                      # метры
        sigma_obs = math.sqrt(float(np.sum(resid ** 2 / sig[:, None] ** 2))
                              / (2 * dof)) if n > 3 else \
            math.sqrt(float(np.mean(resid ** 2)))
        sigma_obs = max(sigma_obs, 1e-6)

        # ---- ковариация параметров (линеаризация, free 8 параметров) ----
        # нормальные уравнения в метровой шкале: N = sum J_i^T W_i J_i,
        # где J_i = dh_метров/dh_free. Ковариация a = pinv(N) (без a-priori).
        free = [0, 1, 2, 3, 4, 5, 6, 7]
        Cxx = np.zeros((8, 8))
        for i, (u, v) in enumerate(pix):
            un, vn = (u - pm[0]) / ps, (v - pm[1]) / ps
            J, _ = _homography_jacobian(Ah, un, vn)
            Ji = J[:, free] * ws                      # переход к метровой шкале
            Cxx += (Ji.T @ Ji) * weights[i]
        try:
            Cxx_inv = np.linalg.pinv(Cxx)
        except np.linalg.LinAlgError:
            Cxx_inv = np.eye(8) * 1e12
        C_h = Cxx_inv                                 # ковариация h в метрах

        model = {
            "Ah": Ah, "pm": pm, "ps": ps, "wm": wm, "ws": ws,
            "origin": origin, "B": B, "lat0": lat0, "lon0": lon0,
            "enu": enu, "pix": pix, "sig": sig, "n": n,
            "resid": resid, "C_h": C_h, "free": free,
            "sigma_obs": sigma_obs, "weights": weights,
        }

        # ---- высота: аффинное пиксель -> up (взвешенный МНК) ------------
        sol_h, Am = _affine_fit(pix, enu[:, 2], weights=weights)
        res_h = enu[:, 2] - Am @ sol_h
        var_h = float(np.sum(res_h ** 2 * weights) / dof) if n > 3 \
            else float(np.mean(res_h ** 2))
        var_h = max(var_h, 1e-12)
        AW = Am * np.sqrt(weights)[:, None]
        C_h_aff = np.linalg.pinv(AW.T @ AW) * var_h
        model.update({"aff_h": sol_h, "cov_h": C_h_aff, "sigma_h": math.sqrt(var_h)})

        model["rmse_horiz"] = math.sqrt(float(np.mean(np.sum(resid ** 2, axis=1))))
        chi2 = float(np.sum(np.sum(resid ** 2, axis=1) / sig ** 2))
        model["chi2_red"] = chi2 / dof
        return model

    # --- предсказание ----------------------------------------------------
    def predict(self, model, u, v):
        """Возвращает (enu_vector, cov3x3_в_ENU) для пикселя (u, v)."""
        Ah, pm, ps, wm, ws = (model["Ah"], model["pm"], model["ps"],
                              model["wm"], model["ws"])
        un = (u - pm[0]) / ps
        vn = (v - pm[1]) / ps
        num = Ah @ np.array([un, vn, 1.0])
        e_n, n_n = num[0] / num[2], num[1] / num[2]
        enu = np.array([e_n * ws + wm[0], n_n * ws + wm[1], 0.0])
        a = np.array([u, v, 1.0])
        enu[2] = float(a @ model["aff_h"])

        # ковариация горизонтальных координат: дельта-метод по 8 параметрам
        free = model["free"]
        J, _ = _homography_jacobian(Ah, un, vn)
        Jm = np.zeros((2, 8))
        Jm[:, free] = J[:, free] * ws                 # метровая шкала
        Cxy = Jm @ model["C_h"] @ Jm.T

        # вклад шума привязки пикселя (sigma_pix) — численная производная
        eps = 1.0
        G = np.zeros((2, 2))
        base = np.array([e_n * ws + wm[0], n_n * ws + wm[1]])
        for ax in range(2):
            du = np.zeros(2)
            du[ax] = eps / ps
            num2 = Ah @ np.array([un + du[0], vn + du[1], 1.0])
            p2 = np.array([num2[0] / num2[2] * ws + wm[0],
                           num2[1] / num2[2] * ws + wm[1]])
            G[:, ax] = (p2 - base) / eps
        sp2 = self.sigma_pix ** 2
        Cxy = Cxy + G @ np.eye(2) * sp2 @ G.T

        # дисперсия высоты: неопределённость коэффициентов + шум пикселя
        grad_h = model["aff_h"][:2]
        var_up = float(a @ model["cov_h"] @ a) + sp2 * float(grad_h @ grad_h)

        cov = np.zeros((3, 3))
        cov[:2, :2] = Cxy
        cov[2, 2] = max(var_up, 1e-12)
        return enu, cov

    def predict_ecef(self, model, u, v):
        enu, C = self.predict(model, u, v)
        B = model["B"]
        xyz = enu_to_ecef(model["origin"], B, enu)
        C_ecef = B.T @ C @ B
        return xyz, C_ecef

    def quality(self, model):
        if model is None:
            return None
        return {"n": model["n"],
                "rmse_horiz_m": model["rmse_horiz"],
                "chi2_red": model["chi2_red"],
                "sigma_up_m": model["sigma_h"]}

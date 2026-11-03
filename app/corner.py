# -*- coding: utf-8 -*-
"""Детектор углов с субпиксельной точностью (Shi–Tomasi + Harris/Скоринг)."""

import numpy as np
import cv2


def _subpixel_refine(gray, cx, cy, win=5):
    """Локальный экстремум корреляции градиентов — субпиксельный сдвиг."""
    h, w = gray.shape
    x0, y0 = int(round(cx)), int(round(cy))
    rx0, ry0 = max(x0 - win, 0), max(y0 - win, 0)
    rx1, ry1 = min(x0 + win + 1, w), min(y0 + win + 1, h)
    roi = gray[ry0:ry1, rx0:rx1].astype(np.float64)
    if roi.size < 9:
        return cx, cy, 0.0
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    # матрица структуры в окне вокруг точки
    iu, iv = y0 - ry0, x0 - rx0
    if not (0 <= iu < roi.shape[0] and 0 <= iv < roi.shape[1]):
        return cx, cy, 0.0
    # весовое гауссово окно
    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    g = np.exp(-((xx - iv) ** 2 + (yy - iu) ** 2) / (2 * (win / 2.0) ** 2))
    a11 = float(np.sum(g * gx * gx))
    a22 = float(np.sum(g * gy * gy))
    a12 = float(np.sum(g * gx * gy))
    det = a11 * a22 - a12 * a12
    trace = a11 + a22
    R = det - 0.04 * trace * trace      # Harris score
    return cx, cy, float(R)


def detect_corner(gray, rect, max_shift=None):
    """Ищет угол внутри прямоугольной зоны `rect=(x0,y0,x1,y1)` (пиксели).

    Возвращает (x_sub, y_sub, strength) с субпиксельными координатами.
    Метод: goodFeaturesToTrack по всей зоне + refinement; если точек нет —
    используется Harris-ответ и максимум по зоне.
    """
    x0, y0, x1, y1 = [int(v) for v in rect]
    x0 = max(x0, 0); y0 = max(y0, 0)
    x1 = min(x1, gray.shape[1]); y1 = min(y1, gray.shape[0])
    if x1 - x0 < 5 or y1 - y0 < 5:
        # слишком маленькая зона — центр
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0
    roi = gray[y0:y1, x0:x1]
    roi_u8 = np.clip(roi, 0, 255).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(roi_u8, maxCorners=5,
                                      qualityLevel=0.01, minDistance=8,
                                      blockSize=7, useHarrisDetector=True, k=0.04)
    best = None
    if corners is not None:
        for c in corners.reshape(-1, 2):
            sx, sy = float(c[0]), float(c[1])
            _, _, R = _subpixel_refine(roi.astype(np.float64), sx, sy)
            if best is None or abs(R) > abs(best[2]):
                best = (sx, sy, R)
    if best is None:
        # Harris-поверхность как резервный вариант
        harris = cv2.cornerHarris(roi_u8.astype(np.float64), 7, 3, 0.04)
        harris = cv2.GaussianBlur(harris, (5, 5), 0)
        idx = np.unravel_index(np.argmax(harris), harris.shape)
        sy, sx = float(idx[1]), float(idx[0])
        _, _, R = _subpixel_refine(roi.astype(np.float64), sx, sy)
        best = (sx, sy, R)
    sx, sy, R = best
    # субпиксельная доводка методом наименьших квадратов по градиентам окна
    sx, sy = _ls_corner_fit(roi.astype(np.float64), sx, sy, win=6)
    return x0 + sx, y0 + sy, R


def _ls_corner_fit(gray, cx, cy, win=6):
    """Подгонка двух ребер (линии Ху–Добсона) вокруг угла; возвращает пересечение."""
    h, w = gray.shape
    ix, iy = int(round(cx)), int(round(cy))
    x0, y0 = max(ix - win, 0), max(iy - win, 0)
    x1, y1 = min(ix + win + 1, w), min(iy + win + 1, h)
    roi = gray[y0:y1, x0:x1]
    if roi.size < 16:
        return cx, cy
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    pts_x = xs.ravel().astype(float) + x0
    pts_y = ys.ravel().astype(float) + y0
    m = mag.ravel()
    gx_f, gy_f = gx.ravel(), gy.ravel()
    keep = m > max(np.percentile(m, 60), 1e-6)
    px, py, nx, ny = pts_x[keep], pts_y[keep], gx_f[keep], gy_f[keep]
    if len(px) < 6:
        return cx, cy
    # Итеративно: разделим пиксели на два кластера по ориентации нормали
    ang = np.arctan2(ny, nx)
    ref = ang[len(ang) // 2]
    d = np.abs(np.angle(np.exp(1j * (ang - ref))))
    mask1 = d < np.pi / 4
    mask2 = ~mask1
    lines = []
    for mk in (mask1, mask2):
        if mk.sum() < 4:
            continue
        X = np.stack([px[mk], py[mk]], axis=1)
        mu = X.mean(axis=0)
        cov = np.cov((X - mu).T)
        evals, evecs = np.linalg.eigh(cov)
        direction = evecs[:, np.argmin(evals)]     # собственное направление меньшей дисперсии
        normal = np.array([-direction[1], direction[0]])
        lines.append((mu, normal))
    if len(lines) < 2:
        return cx, cy
    (m1, n1), (m2, n2) = lines
    A = np.stack([n1, n2], axis=0)
    b = np.array([n1 @ m1, n2 @ m2])
    try:
        inter = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return cx, cy
    if np.linalg.norm(inter - np.array([cx, cy])) > win * 1.5:
        return cx, cy
    return float(inter[0]), float(inter[1])

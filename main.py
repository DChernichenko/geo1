#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка координат по изображению с БПЛА — графический интерфейс (PyQt5).

Запуск:  python main.py

Возможности GUI:
  * открытие растрового изображения, панорамирование колесом/перетаскиванием;
  * выбор системы координат опорных точек (WGS-84, ПЗ-90.27, СК-42 геодезические,
    СК-42/Гаусс-Крюгер, МСК, UTM, произвольная EPSG/смещение геоцентрика);
  * таблица опорных точек: пиксельные и "мировые" координаты, СКО, флаг;
    клик на изображении создаёт точку, двойной клик — редактирование;
  * оценка модели "пиксель -> мир" (гомография + аффинная высота), отчёт о
    качестве (число точек, RMSE, χ², СКО высоты);
  * выделение зоны на изображении (режим «Зона + угол») с автоматическим
    субпиксельным поиском угла (Shi-Tomasi/Harris);
  * расчёт координат отмеченных точек/углов в целевой системе с выводом
    доверительных интервалов 95 % (м и угловые секунды) и ENU;
  * сохранение/загрузка проекта (.cooproj, JSON).
"""

import math
import os
import sys
import traceback

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QColor, QPen, QFont,
                         QKeySequence, QIcon)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QGroupBox, QRadioButton, QButtonGroup, QTextEdit, QAction,
    QFileDialog, QMessageBox, QAbstractItemView, QSizePolicy, QFrame,
    QToolBar, QStatusBar)

from app import geo, project
from app.corner import detect_corner


# ---------------------------------------------------------------------------
# Виджет изображения
# ---------------------------------------------------------------------------
class ImageCanvas(QWidget):
    """Отображение изображения с зумом/панорамой и режимами разметки."""

    clicked = pyqtSignal(float, float)            # пиксельные координаты
    rectSelected = pyqtSignal(float, float, float, float)   # x0,y0,x1,y1

    ZOOM_MIN, ZOOM_MAX = 0.02, 64.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img = None                # QImage для отрисовки
        self._gray = None               # np.uint8 grayscale для детектора
        self._w = self._h = 0
        self._zoom = 1.0
        self._offs = QPointF(0.0, 0.0)  # смещение в экранных координатах
        self.mode = "pick"              # 'pan' | 'pick' | 'rect'
        self.markers = []               # [(u,v,label,color)]
        self.current_rect = None        # рисуемая рамка
        self._dragging = False
        self._press_pos = None          # экранная точка нажатия ЛКМ (режим rect)
        self._drag_start_img = None     # точка начала рамки (изображение)
        self._pan_last = None
        self.setCursor(Qt.CrossCursor)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    # --- данные ----------------------------------------------------------
    def set_image(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._img = qimg
        self._gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._w, self._h = w, h
        self.markers = []
        self.current_rect = None
        self.fit_view()

    def has_image(self):
        return self._img is not None

    @property
    def gray(self):
        return self._gray

    def image_size(self):
        return self._w, self._h

    def set_mode(self, mode):
        self.mode = mode
        self.setCursor(Qt.ArrowCursor if mode == "pan" else Qt.CrossCursor)

    def set_markers(self, markers):
        """markers: список dict(u, v, label, color(QColor|str), kind)."""
        self.markers = list(markers)
        self.update()

    # --- трансформации ---------------------------------------------------
    def fit_view(self):
        if not self.has_image():
            return
        vw, vh = self.width(), self.height()
        z = min(vw / self._w, vh / self._h) * 0.95
        self._zoom = max(self.ZOOM_MIN, min(z, self.ZOOM_MAX))
        self._offs = QPointF((vw - self._w * self._zoom) / 2.0,
                             (vh - self._h * self._zoom) / 2.0)
        self.update()

    def _widget_to_img(self, pos):
        return QPointF((pos.x() - self._offs.x()) / self._zoom,
                       (pos.y() - self._offs.y()) / self._zoom)

    def _img_to_widget(self, p):
        return QPointF(p.x() * self._zoom + self._offs.x(),
                       p.y() * self._zoom + self._offs.y())

    # --- события мыши ----------------------------------------------------
    def wheelEvent(self, ev):
        if not self.has_image():
            return
        factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
        new_zoom = max(self.ZOOM_MIN, min(self._zoom * factor, self.ZOOM_MAX))
        pos = ev.pos()
        img_pt = self._widget_to_img(pos)
        self._zoom = new_zoom
        self._offs = QPointF(pos.x() - img_pt.x() * self._zoom,
                             pos.y() - img_pt.y() * self._zoom)
        self.update()

    def mousePressEvent(self, ev):
        if not self.has_image():
            return
        if ev.button() == Qt.MiddleButton or \
           (ev.button() == Qt.LeftButton and self.mode == "pan") or \
           (ev.button() == Qt.LeftButton and ev.modifiers() & Qt.AltModifier):
            self._pan_last = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if ev.button() == Qt.LeftButton:
            ip = self._widget_to_img(ev.pos())
            # щелчок левой кнопкой мыши в любом режиме разметки
            # ("pick"/"mark") порождает событие clicked; координаты
            # ограничиваются полем изображения
            if self.mode in ("pick", "mark"):
                u = max(0.0, min(ip.x(), self._w - 1.0))
                v = max(0.0, min(ip.y(), self._h - 1.0))
                self.clicked.emit(u, v)
            elif self.mode == "rect":
                self._drag_start_img = ip
                self._press_pos = ev.pos()
                self.current_rect = QRectF(ip, ip)
                self.update()

    def mouseMoveEvent(self, ev):
        if self._pan_last is not None:
            d = ev.pos() - self._pan_last
            self._offs += QPointF(d)
            self._pan_last = ev.pos()
            self.update()
        elif self._drag_start_img is not None:
            # рамка рисуется только после реального перетаскивания (>3 px),
            # чтобы одиночный клик не порождал вырожденную зону
            if (ev.pos() - self._press_pos).manhattanLength() > 3:
                cur = self._widget_to_img(ev.pos())
                self.current_rect = QRectF(self._drag_start_img,
                                           cur).normalized()
                self.update()

    def mouseReleaseEvent(self, ev):
        if self._pan_last is not None:
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor if self.mode == "pan"
                            else Qt.CrossCursor)
        elif self._drag_start_img is not None:
            start = self._drag_start_img
            r = self.current_rect
            dragged = ((ev.pos() - self._press_pos).manhattanLength() > 3
                       and r is not None
                       and r.width() >= 3 and r.height() >= 3)
            self._drag_start_img = None
            self._press_pos = None
            if dragged:
                self.rectSelected.emit(r.left(), r.top(),
                                       r.right(), r.bottom())
            else:
                # простой клик без протяжки — отмечаем пиксель
                u = max(0.0, min(start.x(), self._w - 1.0))
                v = max(0.0, min(start.y(), self._h - 1.0))
                self.clicked.emit(u, v)
            self.current_rect = None
            self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.update()

    # --- отрисовка -------------------------------------------------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(35, 35, 40))
        if not self.has_image():
            p.setPen(QColor(160, 160, 170))
            f = QFont(); f.setPointSize(13); p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Откройте изображение\n(Файл → Открыть изображение…)")
            p.end()
            return
        source = QRectF(0.0, 0.0, float(self._w), float(self._h))
        target = QRectF(self._offs.x(), self._offs.y(),
                        self._w * self._zoom, self._h * self._zoom)
        p.drawPixmap(target, QPixmap.fromImage(self._img), source)
        p.setRenderHint(QPainter.Antialiasing)
        # маркеры: опорные точки — жёлтые полупрозрачные 3x3 пикселя,
        # оценочные точки (пиксели/углы) — синие полупрозрачные 3x3 пикселя
        for m in self.markers:
            wp = self._img_to_widget(QPointF(m["u"], m["v"]))
            kind = m.get("kind", "cp")
            base = QColor(m.get("color", "blue" if kind != "cp" else "yellow"))
            # размер маркера: 3x3 пикселя изображения (в координатах виджета)
            s = max(3.0, 3.0 * self._zoom)
            rect = QRectF(wp.x() - s / 2.0, wp.y() - s / 2.0, s, s)
            fill = QColor(base); fill.setAlpha(140)          # полупрозрачная заливка
            edge = QColor(base); edge.setAlpha(220)          # контур
            p.setPen(QPen(edge, 1))
            p.setBrush(fill)
            p.drawRect(rect)
            # тонкие штрихи-указатели вокруг маркера
            p.setPen(QPen(edge, 1))
            gap = s / 2.0 + 3.0
            ext = gap + 4.0
            p.drawLine(QPointF(wp.x() - ext, wp.y()), QPointF(wp.x() - gap, wp.y()))
            p.drawLine(QPointF(wp.x() + gap, wp.y()), QPointF(wp.x() + ext, wp.y()))
            p.drawLine(QPointF(wp.x(), wp.y() - ext), QPointF(wp.x(), wp.y() - gap))
            p.drawLine(QPointF(wp.x(), wp.y() + gap), QPointF(wp.x(), wp.y() + ext))
            lbl = m.get("label", "")
            if lbl:
                p.setPen(QColor(255, 255, 160))
                f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
                p.drawText(wp + QPointF(gap + 5, -gap - 1), str(lbl))
        # текущая рамка
        if self.current_rect is not None:
            pen = QPen(QColor(80, 200, 255), 1.5, Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            tl = self._img_to_widget(self.current_rect.topLeft())
            br = self._img_to_widget(self.current_rect.bottomRight())
            p.drawRect(QRectF(tl, br).normalized())
        p.end()


# ---------------------------------------------------------------------------
# Диалог координат опорной точки
# ---------------------------------------------------------------------------
class ControlPointDialog(QDialog):
    def __init__(self, converter, u, v, cp=None, parent=None):
        super().__init__(parent)
        self.converter = converter
        self.setWindowTitle("Координаты опорной точки")
        self.resize(420, 210)
        l1, l2, l3 = converter.labels()
        form = QFormLayout()
        self.ed_u = QLineEdit("%.3f" % u)
        self.ed_v = QLineEdit("%.3f" % v)
        self.ed_c1 = QLineEdit()
        self.ed_c2 = QLineEdit()
        self.ed_h = QLineEdit("0.0")
        self.ed_sig = QLineEdit("0.5")
        self.cb_valid = QCheckBox("Использовать точку")
        self.cb_valid.setChecked(True)
        form.addRow("Пиксель u, px:", self.ed_u)
        form.addRow("Пиксель v, px:", self.ed_v)
        form.addRow(l1 + ":", self.ed_c1)
        form.addRow(l2 + ":", self.ed_c2)
        form.addRow(l3 + ":", self.ed_h)
        form.addRow("СКО точки, м:", self.ed_sig)
        form.addRow("", self.cb_valid)
        if cp:
            self.ed_u.setText("%.3f" % cp["u"])
            self.ed_v.setText("%.3f" % cp["v"])
            self.ed_c1.setText(geo.format_dms(cp["c1"], True)
                               if converter.sys.kind == "geodetic"
                               else "%.3f" % cp["c1"])
            self.ed_c2.setText(geo.format_dms(cp["c2"], False)
                               if converter.sys.kind == "geodetic"
                               else "%.3f" % cp["c2"])
            self.ed_h.setText("%.3f" % cp.get("h", 0.0))
            self.ed_sig.setText("%.3f" % cp.get("sigma", 0.5))
            self.cb_valid.setChecked(bool(cp.get("valid", True)))
        btns = QHBoxLayout()
        ok = QPushButton("OK"); ok.setDefault(True)
        cancel = QPushButton("Отмена")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btns.addStretch(1); btns.addWidget(ok); btns.addWidget(cancel)
        lay = QVBoxLayout(self); lay.addLayout(form); lay.addLayout(btns)

    def values(self):
        kind = self.converter.sys.kind
        c1 = (geo.parse_angle(self.ed_c1.text()) if kind == "geodetic"
              else float(self.ed_c1.text().replace(",", ".")))
        c2 = (geo.parse_angle(self.ed_c2.text()) if kind == "geodetic"
              else float(self.ed_c2.text().replace(",", ".")))
        return {"u": float(self.ed_u.text().replace(",", ".")),
                "v": float(self.ed_v.text().replace(",", ".")),
                "c1": c1, "c2": c2,
                "h": float(self.ed_h.text().replace(",", ".")),
                "sigma": float(self.ed_sig.text().replace(",", ".")),
                "valid": self.cb_valid.isChecked()}


# ---------------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Оценка координат по изображению с БПЛА")
        self.resize(1280, 800)

        self.image_path = None
        self.bgr = None
        self.control_points = []          # dicts u,v,c1,c2,h,sigma,valid
        self.marked_points = []           # dicts u,v,kind,rect,corner_strength,result
        self.selected_cp_row = -1
        self.estimator = geo.HomographyEstimator(sigma_pix=0.5)
        self.model = None
        self.custom_system = geo.CustomSystem()

        self._build_ui()
        self._build_actions()
        self._refresh_converter()
        self.statusBar().showMessage("Готово. Откройте изображение.")

    # ======================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # ------- левая часть: изображение -------
        left = QWidget()
        lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas()
        self.canvas.clicked.connect(self._canvas_clicked)
        self.canvas.rectSelected.connect(self._canvas_rect_selected)
        lv.addWidget(self.canvas, 4)
        hint = QLabel("Режим: выберите инструмент на панели сверху. "
                      "Колесо — масштаб, перетаскивание средней кнопкой/Alt — панорама.")
        hint.setStyleSheet("color:#666;")
        lv.addWidget(hint)
        splitter.addWidget(left)

        # ------- правая часть: панель -------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)

        # система координат
        gb_sys = QGroupBox("Система координат")
        sf = QFormLayout(gb_sys)
        self.cb_sys = QComboBox()
        for s in geo.SYSTEMS:
            self.cb_sys.addItem(s.title, s.key)
        self.cb_sys.currentIndexChanged.connect(self._system_changed)
        self.sp_zone = QSpinBox()
        self.sp_zone.setRange(0, 60)          # 0 = авто (по долготе)
        self.sp_zone.setValue(37)
        self.sp_zone.setSpecialValueText("авто")
        self.sp_zone.valueChanged.connect(self._params_changed)
        self.ed_scale = QLineEdit("1.0")
        self.ed_scale.textEdited.connect(self._params_changed)
        self.ed_epsg = QLineEdit()
        self.ed_epsg.setPlaceholderText("напр. EPSG:3857 или код без префикса")
        self.ed_epsg.textEdited.connect(self._params_changed)
        self.lbl_cm = QLabel("")
        sf.addRow("Система:", self.cb_sys)
        sf.addRow("Номер зоны:", self.sp_zone)
        sf.addRow("Масштабный множитель k:", self.ed_scale)
        sf.addRow("Произвольный EPSG:", self.ed_epsg)
        sf.addRow("", self.lbl_cm)
        rv.addWidget(gb_sys)

        # опорные точки
        gb_cp = QGroupBox("Опорные точки")
        cv = QVBoxLayout(gb_cp)
        self.tbl = QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(
            ["u, px", "v, px", "C1", "C2", "H", "СКО", "✓"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.cellChanged.connect(self._table_edited)
        self.tbl.itemSelectionChanged.connect(self._table_sel_changed)
        cv.addWidget(self.tbl, 2)
        row = QHBoxLayout()
        self.btn_add_cp = QPushButton("Добавить…")
        self.btn_del_cp = QPushButton("Удалить")
        self.btn_edit_cp = QPushButton("Изменить…")
        self.btn_add_cp.clicked.connect(self._add_cp_manual)
        self.btn_del_cp.clicked.connect(self._delete_cp)
        self.btn_edit_cp.clicked.connect(self._edit_selected_cp)
        for b in (self.btn_add_cp, self.btn_edit_cp, self.btn_del_cp):
            row.addWidget(b)
        cv.addLayout(row)
        rv.addWidget(gb_cp, 3)

        # режимы разметки
        gb_mode = QGroupBox("Инструменты")
        mv = QVBoxLayout(gb_mode)
        self.bg = QButtonGroup(self)
        rb_pan = QRadioButton("Панорама")
        rb_pick = QRadioButton("Точка опоры (клик по изображению)")
        rb_pick.setChecked(True)
        rb_rect = QRadioButton("Зона + угол (выделить рамкой)")
        rb_mark = QRadioButton("Отметить пиксель (оценка координат)")
        for i, rb in enumerate((rb_pan, rb_pick, rb_rect, rb_mark)):
            self.bg.addButton(rb, i); mv.addWidget(rb)
            rb.toggled.connect(lambda ch, k=i: ch and self._mode_changed(k))
        self.lbl_zone = QLabel("Зона не выбрана")
        self.lbl_zone.setWordWrap(True)
        mv.addWidget(self.lbl_zone)
        rv.addWidget(gb_mode)

        # результат
        gb_res = QGroupBox("Результат оценки")
        rvl = QVBoxLayout(gb_res)
        self.txt_quality = QLabel("Модель не рассчитана")
        self.txt_quality.setWordWrap(True)
        rvl.addWidget(self.txt_quality)
        self.btn_calc = QPushButton("Рассчитать модель и координаты")
        self.btn_calc.clicked.connect(self._compute)
        rvl.addWidget(self.btn_calc)
        self.txt_results = QTextEdit()
        self.txt_results.setReadOnly(True)
        self.txt_results.setFont(QFont("Consolas", 9))
        rvl.addWidget(self.txt_results, 2)
        rv.addWidget(gb_res, 3)

        splitter.addWidget(right)
        splitter.setSizes([860, 420])

    def _build_actions(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&Файл")
        act_open_img = QAction("Открыть &изображение…", self,
                              shortcut=QKeySequence.Open, triggered=self._open_image)
        act_save = QAction("&Сохранить проект…", self,
                           shortcut=QKeySequence.Save, triggered=self._save_project)
        act_saveas = QAction("Сохранить проект &как…", self,
                             shortcut=QKeySequence("Ctrl+Shift+S"),
                             triggered=lambda: self._save_project(as_new=True))
        act_open_prj = QAction("Открыть &проект…", self,
                               shortcut=QKeySequence("Ctrl+Shift+O"),
                               triggered=self._open_project)
        act_quit = QAction("Вы&ход", self, shortcut="Ctrl+Q",
                           triggered=self.close)
        for a in (act_open_img, None, act_save, act_saveas, act_open_prj,
                  None, act_quit):
            if a is None:
                m_file.addSeparator()
            else:
                m_file.addAction(a)
        m_help = mb.addMenu("&Справка")
        m_help.addAction(QAction("О программе", self, triggered=self._about))

        tb = QToolBar("_main")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(act_open_img)
        tb.addAction(act_save)
        tb.addAction(act_open_prj)
        tb.addSeparator()
        tb.addAction(QAction("Рассчитать", self, triggered=self._compute))
        tb.addSeparator()
        zoom_in = QAction("＋", self, triggered=lambda: self._zoom_step(1.25))
        zoom_out = QAction("－", self, triggered=lambda: self._zoom_step(0.8))
        zoom_fit = QAction("⤢ Вписать", self, triggered=self.canvas.fit_view)
        tb.addAction(zoom_in); tb.addAction(zoom_out); tb.addAction(zoom_fit)

    # ======================================================================
    # Конвертер активной системы
    # ======================================================================
    def _current_sys_key(self):
        return self.cb_sys.currentData()

    def _refresh_converter(self):
        key = self._current_sys_key()
        zone = self.sp_zone.value() if self.sp_zone.value() > 0 else None
        try:
            scale = float(self.ed_scale.text().replace(",", "."))
        except ValueError:
            scale = None
        epsg_txt = self.ed_epsg.text().strip()
        if key == "custom":
            self.custom_system = geo.CustomSystem("Пользовательская",
                                                  offset=(0, 0, 0),
                                                  epsg=epsg_txt or None)
        self.converter = geo.GeoConverter(key, zone=zone, scale=scale,
                                          custom=self.custom_system)
        # доступность параметров зоны/масштаба/EPSG
        is_gauss = self.converter.sys.kind == "gauss"
        self.sp_zone.setEnabled(is_gauss)
        self.ed_scale.setEnabled(is_gauss and key == "msk")
        self.ed_epsg.setEnabled(key == "custom")
        if is_gauss:
            cm = self.converter.central_meridian
            self.lbl_cm.setText("ЦМ зоны: %s°%s" % (
                ("%.0f" % cm) if zone else "авто",
                "" if zone else " (по долготе результата)"))
        else:
            self.lbl_cm.setText("")

    def _system_changed(self, *_):
        self._refresh_converter()

    def _params_changed(self, *_):
        self._refresh_converter()

    # ======================================================================
    # Изображение
    # ======================================================================
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть изображение", "",
            "Изображения (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp);;Все файлы (*)")
        if not path:
            return
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось прочитать изображение:\n" + path)
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.bgr = img
        self.image_path = path
        self.canvas.set_image(img)
        self.setWindowTitle("Оценка координат по изображению с БПЛА — " +
                            os.path.basename(path))
        self.statusBar().showMessage(
            "Изображение: %dx%d px" % (img.shape[1], img.shape[0]))
        self._update_canvas_markers()

    def _zoom_step(self, factor):
        # центрированный зум относительно центра виджета
        cx, cy = self.canvas.width() / 2.0, self.canvas.height() / 2.0
        old = self.canvas._zoom
        new = max(ImageCanvas.ZOOM_MIN, min(old * factor, ImageCanvas.ZOOM_MAX))
        img_pt = self.canvas._widget_to_img(QPointF(cx, cy))
        self.canvas._zoom = new
        self.canvas._offs = QPointF(cx - img_pt.x() * new, cy - img_pt.y() * new)
        self.canvas.update()

    # ======================================================================
    # Режимы и разметка
    # ======================================================================
    def _mode_changed(self, idx):
        self.canvas.set_mode({0: "pan", 1: "pick", 2: "rect", 3: "mark"}[idx])

    def _canvas_clicked(self, u, v):
        """Обработка щелчка ЛКМ по изображению (сигнал canvas.clicked).

        Режим "mark" — добавление пикселя для оценки координат;
        режим "rect" — одиночный клик (без протяжки) также отмечает пиксель;
        режим "pick" — создание опорной точки.
        """
        if not self.canvas.has_image():
            return
        w, h = self.canvas.image_size()
        u = max(0.0, min(u, w - 1.0)); v = max(0.0, min(v, h - 1.0))
        mode = self.canvas.mode
        if mode == "mark":
            # ключевое исправление: отметка пикселя по щелчку ЛКМ
            self._add_marked_point(u, v, "pixel")
            return
        if mode == "rect":
            # одиночный клик в режиме зоны — отмечаем конкретный пиксель
            self._add_marked_point(u, v, "pixel")
            return
        if mode != "pick":
            return
        dlg = ControlPointDialog(self.converter, u, v, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            try:
                vals = dlg.values()
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "Некорректный ввод координат.")
                return
            self.control_points.append(vals)
            self._sync_table_from_data()
            self._update_canvas_markers()
            self.statusBar().showMessage(
                "Добавлена опорная точка №%d (%.1f; %.1f)" %
                (len(self.control_points), u, v))

    def _canvas_rect_selected(self, x0, y0, x1, y1):
        """Режим «Зона + угол»: автодетект угла в рамке."""
        if self.canvas.gray is None:
            return
        gray = self.canvas.gray
        strength = None
        try:
            cx, cy, R = detect_corner(gray, (x0, y0, x1, y1))
            strength = float(R)
        except Exception:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        self.lbl_zone.setText(
            "Зона (%.0f, %.0f)–(%.0f, %.0f); угол: (%.2f; %.2f)%s" %
            (x0, y0, x1, y1, cx, cy,
             ", Harris=%.3g" % strength if strength is not None else ""))
        self._add_marked_point(cx, cy, "corner", rect=(x0, y0, x1, y1),
                               strength=strength)

    def _add_marked_point(self, u, v, kind, rect=None, strength=None):
        mp = {"u": u, "v": v, "kind": kind, "rect": rect,
              "corner_strength": strength, "result": None}
        self.marked_points.append(mp)
        # если модель ещё не рассчитана, но точек достаточно — рассчитать
        if self.model is None and \
           len(self._control_ecef_list()) >= self.estimator.MIN_POINTS:
            try:
                self.model = self.estimator.fit(self._control_ecef_list())
                q = self.estimator.quality(self.model)
                self.txt_quality.setText(
                    "N=%d  RMSEгориз=%.3f м  χ²/доф=%.2f  σвыс=%.3f м" %
                    (q["n"], q["rmse_horiz_m"], q["chi2_red"], q["sigma_up_m"]))
            except Exception:
                self.model = None
        self._compute_single(mp)
        self._update_canvas_markers()
        self._render_results()
        res = mp.get("result")
        if res:
            self.statusBar().showMessage(
                "Отмечен пиксель (%.2f; %.2f), координаты оценены." % (u, v))
        else:
            self.statusBar().showMessage(
                "Отмечен пиксель (%.2f; %.2f). Для оценки координат нужно "
                "не менее %d опорных точек и расчёт модели." %
                (u, v, self.estimator.MIN_POINTS))

    # ======================================================================
    # Таблица опорных точек
    # ======================================================================
    def _fmt_c(self, value):
        if self.converter.sys.kind == "geodetic":
            return geo.format_dms(value, True)
        return "%.3f" % value

    def _sync_table_from_data(self):
        self.tbl.blockSignals(True)
        kind = self.converter.sys.kind
        l1, l2, l3 = self.converter.labels()
        labels = ["u, px", "v, px", l1, l2, l3, "СКО, м", "✓"]
        self.tbl.setHorizontalHeaderLabels(labels)
        self.tbl.setRowCount(len(self.control_points))
        for r, p in enumerate(self.control_points):
            cells = [
                "%.3f" % p["u"], "%.3f" % p["v"],
                (geo.format_dms(p["c1"], True) if kind == "geodetic"
                 else "%.3f" % p["c1"]),
                (geo.format_dms(p["c2"], False) if kind == "geodetic"
                 else "%.3f" % p["c2"]),
                "%.3f" % p.get("h", 0.0),
                "%.3f" % p.get("sigma", 0.5),
            ]
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if c == 6:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.tbl.setItem(r, c, it)
            it = QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled |
                        Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if p.get("valid", True) else Qt.Unchecked)
            self.tbl.setItem(r, 6, it)
        self.tbl.blockSignals(False)

    def _table_edited(self, row, col):
        if self.tbl.item(row, col) is None:
            return
        item = self.tbl.item(row, col)
        if col == 6:
            self.control_points[row]["valid"] = (
                item.checkState() == Qt.Checked)
            self._update_canvas_markers()
            return
        if not (0 <= row < len(self.control_points)):
            return
        txt = item.text()
        try:
            if col in (2, 3) and self.converter.sys.kind == "geodetic":
                val = geo.parse_angle(txt)
            else:
                val = float(txt.replace(",", "."))
        except ValueError:
            self.statusBar().showMessage("Не удалось разобрать значение: %r" % txt)
            return
        keys = ["u", "v", "c1", "c2", "h", "sigma"]
        self.control_points[row][keys[col]] = val
        self._update_canvas_markers()

    def _table_sel_changed(self):
        rows = {i.row() for i in self.tbl.selectedItems()}
        self.selected_cp_row = min(rows) if rows else -1

    def _add_cp_manual(self):
        u = v = 0.0
        if self.canvas.has_image():
            pass  # пусть пользователь введёт сам
        dlg = ControlPointDialog(self.converter, u, v, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            try:
                self.control_points.append(dlg.values())
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "Некорректный ввод.")
                return
            self._sync_table_from_data()
            self._update_canvas_markers()

    def _delete_cp(self):
        if self.selected_cp_row < 0:
            QMessageBox.information(self, "Удаление", "Выберите строку таблицы.")
            return
        del self.control_points[self.selected_cp_row]
        self.selected_cp_row = -1
        self._sync_table_from_data()
        self._update_canvas_markers()

    def _edit_selected_cp(self):
        if self.selected_cp_row < 0:
            QMessageBox.information(self, "Изменение", "Выберите строку таблицы.")
            return
        p = self.control_points[self.selected_cp_row]
        dlg = ControlPointDialog(self.converter, p["u"], p["v"], cp=p, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            try:
                self.control_points[self.selected_cp_row] = dlg.values()
            except ValueError:
                return
            self._sync_table_from_data()
            self._update_canvas_markers()

    # ======================================================================
    # Маркеры на холсте
    # ======================================================================
    def _update_canvas_markers(self):
        ms = []
        for i, p in enumerate(self.control_points):
            # опорные точки: жёлтый полупрозрачный маркер 3x3 пикселя
            ms.append({"u": p["u"], "v": p["v"], "label": "ОП%d" % (i + 1),
                       "kind": "cp",
                       "color": "#ffe000" if p.get("valid", True) else "#888888"})
        for j, m in enumerate(self.marked_points):
            res = m.get("result")
            # оценочные точки: синий полупрозрачный маркер 3x3 пикселя
            col = "#2060ff" if res else "#70a0d0"
            ms.append({"u": m["u"], "v": m["v"],
                       "label": ("У%d" if m["kind"] == "corner" else "П%d") % (j + 1),
                       "kind": m["kind"], "color": col})
        self.canvas.set_markers(ms)

    # ======================================================================
    # Вычисления
    # ======================================================================
    def _control_ecef_list(self):
        pts = []
        for p in self.control_points:
            if not p.get("valid", True):
                continue
            try:
                ecef = self.converter.world_to_ecef(p["c1"], p["c2"], p["h"])
            except Exception:
                continue
            pts.append({"u": p["u"], "v": p["v"], "ecef": ecef,
                        "sigma_obj": p.get("sigma", 0.5), "valid": True})
        return pts

    def _compute(self):
        if not self.canvas.has_image():
            QMessageBox.information(self, "Нет изображения",
                                    "Сначала откройте изображение.")
            return
        pts = self._control_ecef_list()
        if len(pts) < self.estimator.MIN_POINTS:
            QMessageBox.warning(
                self, "Мало точек",
                "Для оценки модели нужно не менее %d активных опорных точек."
                % self.estimator.MIN_POINTS)
            return
        try:
            self.model = self.estimator.fit(pts)
        except Exception as ex:
            QMessageBox.critical(self, "Ошибка оценки", str(ex))
            return
        if self.model is None:
            QMessageBox.critical(self, "Ошибка", "Модель не построена.")
            return
        q = self.estimator.quality(self.model)
        self.txt_quality.setText(
            "N=%d  RMSEгориз=%.3f м  χ²/доф=%.2f  σвыс=%.3f м" %
            (q["n"], q["rmse_horiz_m"], q["chi2_red"], q["sigma_up_m"]))
        for m in self.marked_points:
            self._compute_single(m)
        self._update_canvas_markers()
        self._render_results()
        self.statusBar().showMessage("Модель рассчитана.")

    def _compute_single(self, mp):
        if self.model is None:
            mp["result"] = None
            return
        try:
            enu, cov = self.estimator.predict(self.model, mp["u"], mp["v"])
            xyz, _ = self.estimator.predict_ecef(self.model, mp["u"], mp["v"])
            lat, lon, h = geo.ecef_to_geodetic(xyz)
            world = self.converter.ecef_to_world(xyz)
            ci_diag = 1.96 * np.sqrt(np.clip(np.diag(cov), 0, None))
            # ДИ в широту/долготу (секунды дуги) и высоту (м)
            a_axis = geo.WGS84_A
            e2 = geo.WGS84_F * (2 - geo.WGS84_F)
            sl = math.sin(math.radians(lat))
            M = a_axis * (1 - e2) / (1 - e2 * sl * sl) ** 1.5
            Np = a_axis / math.sqrt(1 - e2 * sl * sl)
            sec = 180.0 / math.pi * 3600.0
            cos_lat = max(abs(math.cos(math.radians(lat))), 1e-9)
            ci_latlon = (ci_diag[1] / M * sec,
                         ci_diag[0] / (Np * cos_lat) * sec,
                         ci_diag[2])
            mp["result"] = {
                "world": [float(world[0]), float(world[1]), float(world[2])],
                "latlonh": [float(lat), float(lon), float(h)],
                "enu": [float(x) for x in enu],
                "ci_world": [float(x) for x in ci_diag],
                "ci_latlon_m": [float(x) for x in ci_latlon],
            }
        except Exception:
            mp["result"] = None

    def _render_results(self):
        lines = []
        kind = self.converter.sys.kind
        title = self.converter.sys.title
        lines.append("Целевая система: %s" % title)
        lines.append("Доверительный интервал 95 %% (k=1.96)\n")
        if not self.marked_points:
            lines.append("Отмеченных точек нет. Используйте инструменты "
                         "«Отметить пиксель» или «Зона + угол».")
        for j, m in enumerate(self.marked_points):
            tag = ("Угол У%d" % (j + 1)) if m["kind"] == "corner" \
                else ("Пиксель П%d" % (j + 1))
            lines.append("%s  (u=%.2f; v=%.2f)%s" %
                         (tag, m["u"], m["v"],
                          ", Harris=%.3g" % m["corner_strength"]
                          if m.get("corner_strength") is not None else ""))
            r = m.get("result")
            if not r:
                lines.append("   — нет модели: нажмите «Рассчитать».\n")
                continue
            w = r["world"]; cw = r["ci_world"]
            if kind == "geodetic":
                lines.append("   B=%s  L=%s  H=%.3f м" %
                             (geo.format_dms(w[0], True),
                              geo.format_dms(w[1], False), w[2]))
                cl = r["ci_latlon_m"]
                lines.append("   ДИ: широта ±%.1f″  долгота ±%.1f″  высота ±%.3f м"
                             % (cl[0], cl[1], cl[2]))
            else:
                lines.append("   %s=%.3f  %s=%.3f  H=%.3f м" %
                             (self.converter.labels()[0], w[0],
                              self.converter.labels()[1], w[1], w[2]))
                lines.append("   ДИ: C1 ±%.3f м  C2 ±%.3f м  H ±%.3f м"
                             % tuple(cw))
            e = r["latlonh"]
            lines.append("   WGS-84: B=%s L=%s H=%.3f м" %
                         (geo.format_dms(e[0], True), geo.format_dms(e[1], False), e[2]))
            enu = r["enu"]
            lines.append("   ENU (отн. первой ОП): E=%.3f  N=%.3f  U=%.3f м\n"
                         % (enu[0], enu[1], enu[2]))
        self.txt_results.setPlainText("\n".join(lines))

    # ======================================================================
    # Проект
    # ======================================================================
    def _save_project(self, as_new=False):
        if not self.image_path:
            QMessageBox.information(self, "Проект", "Нет изображения — нечего сохранять.")
            return
        if as_new or not getattr(self, "project_path", None):
            path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить проект", "project" + project.PROJECT_EXT,
                "Проекты (*%s);;JSON (*.json)" % project.PROJECT_EXT)
            if not path:
                return
            if not path.lower().endswith(project.PROJECT_EXT):
                path += project.PROJECT_EXT
            self.project_path = path
        zone = self.sp_zone.value() if self.sp_zone.value() > 0 else None
        try:
            scale = float(self.ed_scale.text().replace(",", "."))
        except ValueError:
            scale = None
        project.save_project(self.project_path, self.image_path,
                             self._current_sys_key(), zone, scale,
                             self.ed_epsg.text().strip() or None,
                             self.control_points, self.marked_points)
        self.statusBar().showMessage("Проект сохранён: " + self.project_path)

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть проект", "",
            "Проекты (*%s);;JSON (*.json);;Все файлы (*)" % project.PROJECT_EXT)
        if not path:
            return
        self._load_project_path(path)

    def _load_project_path(self, path):
        try:
            data = project.load_project(path)
        except Exception as ex:
            QMessageBox.critical(self, "Ошибка загрузки", str(ex))
            return
        img = data["image"]
        if img and os.path.isfile(img):
            self._load_image_from_file(img)
        else:
            QMessageBox.warning(self, "Проект",
                                "Изображение из проекта не найдено:\n%s" % img)
        idx = self.cb_sys.findData(data["sys_key"])
        if idx >= 0:
            self.cb_sys.setCurrentIndex(idx)
        self.sp_zone.setValue(int(data["zone"]) if data["zone"] else 0)
        self.ed_scale.setText("" if data["scale"] is None else str(data["scale"]))
        self.ed_epsg.setText(data.get("custom_epsg") or "")
        self._refresh_converter()
        self.control_points = data["control_points"]
        self.marked_points = data["marked_points"]
        self.project_path = path
        self._sync_table_from_data()
        self._update_canvas_markers()
        if len(self._control_ecef_list()) >= self.estimator.MIN_POINTS:
            self._compute()
        self.statusBar().showMessage("Проект загружен: " + path)

    def _load_image_from_file(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.bgr = img
        self.image_path = path
        self.canvas.set_image(img)

    # ======================================================================
    def _about(self):
        QMessageBox.about(
            self, "О программе",
            "Оценка координат по изображению с БПЛА.\n\n"
            "Метод: привязка опорных точек (гомография, МНК) + аффинная\n"
            "модель высот; распространение ошибок дельта-методом,\n"
            "доверительные интервалы 95 %. Пересчёты СК: Гельмерт,\n"
            "Гаусс–Крюгер (ряды), локальная ENU.")


def excepthook(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    sys.stderr.write(msg)
    try:
        QMessageBox.critical(None, "Ошибка", msg[-2000:])
    except Exception:
        pass


def main():
    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    # поддержка открытия файла аргументом командной строки
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower().endswith(project.PROJECT_EXT) or arg.endswith(".json"):
            win._open_project_path(arg)
        elif os.path.isfile(arg):
            win._load_image_from_file(arg)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

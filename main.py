import csv
import random
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


MODEL_NAME = "yolo11n.pt"
PERSON_CLASS_ID = 0
CONFIDENCE = 0.35

LINE_POSITION = 0.55
LINE_MARGIN = 18

SIM_WIDTH = 960
SIM_HEIGHT = 540
SIM_PERSON_HEIGHT = 210

SIM_TIMER_MS = 25
VIDEO_TIMER_MS = 1

MANUAL_SPEED_MIN = 7.0
MANUAL_SPEED_MAX = 9.0

AUTO_TEST_SPEED_MIN = 10.0
AUTO_TEST_SPEED_MAX = 12.0

AUTO_TEST_PLAN = [
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
]

AUTO_TEST_GAP_FRAMES = 8
AUTO_TEST_FINISH_FRAMES = 15


@dataclass
class SimPerson:
    x: float
    y: float
    direction: int
    speed: float
    height: int = SIM_PERSON_HEIGHT


class DatabaseManager:

    def __init__(self, path: Path):
        self.path = path
        self._init_db()

    def connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self):

        with self.connect() as con:

            con.execute(
                """
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )

            con.execute(
                """
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_time TEXT NOT NULL,
                    expected_in INTEGER NOT NULL,
                    expected_out INTEGER NOT NULL,
                    detected_in INTEGER NOT NULL,
                    detected_out INTEGER NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1 REAL NOT NULL,
                    count_accuracy REAL NOT NULL
                )
                """
            )

            con.commit()

    def add_visit(
        self,
        track_id: int,
        event_type: str,
        source: str,
    ):

        event_time = (
            datetime
            .now()
            .replace(microsecond=0)
            .isoformat(sep=" ")
        )

        with self.connect() as con:

            con.execute(
                """
                INSERT INTO visits (
                    event_time,
                    track_id,
                    event_type,
                    source
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    event_time,
                    int(track_id),
                    event_type,
                    source,
                ),
            )

            con.commit()

    def today_visits(self):

        today = date.today().isoformat()

        with self.connect() as con:

            return con.execute(
                """
                SELECT
                    event_time,
                    track_id,
                    event_type,
                    source
                FROM visits
                WHERE date(event_time) = ?
                ORDER BY id
                """,
                (today,),
            ).fetchall()

    def clear_today_visits(self):

        today = date.today().isoformat()

        with self.connect() as con:

            con.execute(
                """
                DELETE FROM visits
                WHERE date(event_time) = ?
                """,
                (today,),
            )

            con.commit()

    def save_test(
        self,
        expected_in,
        expected_out,
        detected_in,
        detected_out,
        precision,
        recall,
        f1,
        count_accuracy,
    ):

        run_time = (
            datetime
            .now()
            .replace(microsecond=0)
            .isoformat(sep=" ")
        )

        with self.connect() as con:

            con.execute(
                """
                INSERT INTO test_results (
                    run_time,
                    expected_in,
                    expected_out,
                    detected_in,
                    detected_out,
                    precision,
                    recall,
                    f1,
                    count_accuracy
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_time,
                    expected_in,
                    expected_out,
                    detected_in,
                    detected_out,
                    precision,
                    recall,
                    f1,
                    count_accuracy,
                ),
            )

            con.commit()

    def last_test(self):

        with self.connect() as con:

            return con.execute(
                """
                SELECT
                    run_time,
                    expected_in,
                    expected_out,
                    detected_in,
                    detected_out,
                    precision,
                    recall,
                    f1,
                    count_accuracy
                FROM test_results
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

    def hourly_today(self):

        today = date.today().isoformat()

        with self.connect() as con:

            return con.execute(
                """
                SELECT
                    strftime('%H:00', event_time),

                    SUM(
                        CASE
                            WHEN event_type = 'ВХОД'
                            THEN 1
                            ELSE 0
                        END
                    ),

                    SUM(
                        CASE
                            WHEN event_type = 'ВЫХОД'
                            THEN 1
                            ELSE 0
                        END
                    )

                FROM visits

                WHERE date(event_time) = ?

                GROUP BY strftime('%H', event_time)

                ORDER BY strftime('%H', event_time)
                """,
                (today,),
            ).fetchall()


class AnalyticsDialog(QDialog):

    def __init__(
        self,
        parent,
        db: DatabaseManager,
    ):

        super().__init__(parent)

        self.setWindowTitle(
            "Аналитика посещаемости"
        )

        self.resize(
            650,
            520,
        )

        self.setMinimumSize(
            560,
            430,
        )

        self.setStyleSheet(
            MainWindow.dialog_stylesheet()
        )

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(14)

        title = QLabel(
            "АНАЛИТИКА ПОСЕЩАЕМОСТИ"
        )

        title.setObjectName(
            "analyticsTitle"
        )

        layout.addWidget(title)

        rows = db.hourly_today()

        total_in = sum(
            row[1] or 0
            for row in rows
        )

        total_out = sum(
            row[2] or 0
            for row in rows
        )

        inside = max(
            total_in - total_out,
            0,
        )

        summary = QLabel(
            f"Дата: "
            f"{datetime.now().strftime('%d.%m.%Y')}\n"

            f"Всего входов: "
            f"{total_in}   |   "

            f"Всего выходов: "
            f"{total_out}   |   "

            f"На территории: "
            f"{inside}"
        )

        summary.setObjectName(
            "analyticsSummary"
        )

        layout.addWidget(summary)

        table = QTableWidget(
            0,
            4,
        )

        table.setObjectName(
            "analyticsTable"
        )

        table.setHorizontalHeaderLabels(
            [
                "Час",
                "Входы",
                "Выходы",
                "Баланс",
            ]
        )

        table.verticalHeader().setVisible(
            False
        )

        table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        if rows:

            for (
                hour,
                entered,
                exited,
            ) in rows:

                entered = entered or 0
                exited = exited or 0

                row_index = table.rowCount()

                table.insertRow(
                    row_index
                )

                values = (
                    hour,
                    entered,
                    exited,
                    entered - exited,
                )

                for column, value in enumerate(
                    values
                ):

                    table.setItem(
                        row_index,
                        column,
                        QTableWidgetItem(
                            str(value)
                        ),
                    )

        else:

            table.setRowCount(1)

            item = QTableWidgetItem(
                "За текущий день данных пока нет"
            )

            item.setTextAlignment(
                Qt.AlignCenter
            )

            table.setItem(
                0,
                0,
                item,
            )

            table.setSpan(
                0,
                0,
                1,
                4,
            )

        layout.addWidget(
            table,
            1,
        )

        row = QHBoxLayout()

        row.addStretch()

        close_button = QPushButton(
            "Закрыть"
        )

        close_button.setObjectName(
            "primaryButton"
        )

        close_button.clicked.connect(
            self.accept
        )

        row.addWidget(
            close_button
        )

        layout.addLayout(row)


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.entered = 0
        self.exited = 0

        self.video_capture: Optional[
            cv2.VideoCapture
        ] = None

        self.source_mode: Optional[
            str
        ] = None

        self.track_sides = {}

        self.model: Optional[
            YOLO
        ] = None

        self.person_sprite: Optional[
            np.ndarray
        ] = None

        self.person_sprite_name: Optional[
            str
        ] = None

        self.sim_people: list[
            SimPerson
        ] = []

        self.test_active = False
        self.test_plan: list[int] = []

        self.test_expected_in = 0
        self.test_expected_out = 0

        self.test_detected_in = 0
        self.test_detected_out = 0

        self.test_idle_frames = 0

        self.db_path = (
            Path(__file__)
            .resolve()
            .with_name(
                "visits.db"
            )
        )

        self.db = DatabaseManager(
            self.db_path
        )

        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.process_next_frame
        )

        self.setWindowTitle(
            "Система учёта посещений предприятия"
        )

        self.resize(
            1450,
            900,
        )

        self.setMinimumSize(
            1180,
            760,
        )

        self.build_ui()

        self.refresh_today_data()

        self.refresh_last_test()

        self.load_model()

    def build_ui(self):

        root = QWidget()

        self.setCentralWidget(root)

        main = QVBoxLayout(root)

        main.setContentsMargins(
            24,
            18,
            24,
            18,
        )

        main.setSpacing(14)


        header = QHBoxLayout()

        title_box = QVBoxLayout()

        title = QLabel(
            "СИСТЕМА УЧЁТА ПОСЕЩЕНИЙ ПРЕДПРИЯТИЯ"
        )

        title.setObjectName(
            "title"
        )

        subtitle = QLabel(
            "Компьютерное зрение: "
            "YOLO11n + ByteTrack"
        )

        subtitle.setObjectName(
            "subtitle"
        )

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.status_label = QLabel(
            "● Инициализация нейросети..."
        )

        self.status_label.setObjectName(
            "statusStopped"
        )

        self.status_label.setAlignment(
            Qt.AlignRight
            |
            Qt.AlignVCenter
        )

        header.addLayout(
            title_box,
            1,
        )

        header.addWidget(
            self.status_label
        )

        main.addLayout(header)


        content = QHBoxLayout()

        content.setSpacing(14)

        camera_panel = QFrame()

        camera_panel.setObjectName(
            "panel"
        )

        camera_layout = QVBoxLayout(
            camera_panel
        )

        camera_layout.setContentsMargins(
            16,
            16,
            16,
            16,
        )

        camera_layout.setSpacing(10)

        camera_header = QHBoxLayout()

        camera_title = QLabel(
            "Камера / виртуальная проходная"
        )

        camera_title.setObjectName(
            "sectionTitle"
        )

        self.source_badge = QLabel(
            "Источник не выбран"
        )

        self.source_badge.setObjectName(
            "badge"
        )

        camera_header.addWidget(
            camera_title
        )

        camera_header.addStretch()

        camera_header.addWidget(
            self.source_badge
        )

        self.camera_label = QLabel(
            "Выберите режим работы:\n\n"

            "Открыть видео — "
            "анализ видеофайла\n"

            "Веб-камера — "
            "анализ живого изображения\n"

            "Симулятор — "
            "виртуальная проходная предприятия"
        )

        self.camera_label.setObjectName(
            "camera"
        )

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setMinimumSize(
            760,
            500,
        )

        hint = QLabel(
            "Контрольная линия: "
            "сверху вниз — ВХОД, "
            "снизу вверх — ВЫХОД"
        )

        hint.setObjectName(
            "hint"
        )

        hint.setAlignment(
            Qt.AlignCenter
        )

        camera_layout.addLayout(
            camera_header
        )

        camera_layout.addWidget(
            self.camera_label,
            1,
        )

        camera_layout.addWidget(
            hint
        )

        right_panel = QFrame()

        right_panel.setObjectName(
            "panel"
        )

        right_panel.setMinimumWidth(
            430
        )

        right = QVBoxLayout(
            right_panel
        )

        right.setContentsMargins(
            16,
            16,
            16,
            16,
        )

        right.setSpacing(10)

        stat_title = QLabel(
            "Статистика"
        )

        stat_title.setObjectName(
            "sectionTitle"
        )

        right.addWidget(
            stat_title
        )

        cards = QGridLayout()

        cards.setSpacing(10)

        (
            in_card,
            self.entered_label,
        ) = self.stat_card(
            "ВОШЛО"
        )

        (
            out_card,
            self.exited_label,
        ) = self.stat_card(
            "ВЫШЛО"
        )

        (
            inside_card,
            self.inside_label,
        ) = self.stat_card(
            "НА ТЕРРИТОРИИ"
        )

        cards.addWidget(
            in_card,
            0,
            0,
        )

        cards.addWidget(
            out_card,
            0,
            1,
        )

        cards.addWidget(
            inside_card,
            1,
            0,
            1,
            2,
        )

        right.addLayout(cards)

        self.model_info = QLabel()

        self.model_info.setObjectName(
            "infoCard"
        )

        self.refresh_model_info()

        right.addWidget(
            self.model_info
        )

        self.today_info = QLabel()

        self.today_info.setObjectName(
            "infoCard"
        )

        right.addWidget(
            self.today_info
        )

        test_title = QLabel(
            "Оценка качества"
        )

        test_title.setObjectName(
            "sectionTitle"
        )

        right.addWidget(
            test_title
        )

        self.test_info = QLabel(
            "Автотест ещё не выполнялся.\n"

            "Тестовый сценарий: "
            "10 проходов "
            "(6 входов / 4 выхода)."
        )

        self.test_info.setObjectName(
            "infoCard"
        )

        self.test_info.setWordWrap(True)

        right.addWidget(
            self.test_info
        )

        log_title = QLabel(
            "Журнал событий за сегодня"
        )

        log_title.setObjectName(
            "sectionTitle"
        )

        right.addWidget(
            log_title
        )

        self.events_table = QTableWidget(
            0,
            4,
        )

        self.events_table.setObjectName(
            "eventsTable"
        )

        self.events_table.setHorizontalHeaderLabels(
            [
                "Время",
                "ID",
                "Событие",
                "Источник",
            ]
        )

        self.events_table.verticalHeader().setVisible(
            False
        )

        self.events_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        for column in (
            0,
            1,
            2,
        ):

            self.events_table.horizontalHeader().setSectionResizeMode(
                column,
                QHeaderView.ResizeToContents,
            )

        self.events_table.horizontalHeader().setSectionResizeMode(
            3,
            QHeaderView.Stretch,
        )

        self.events_table.setMinimumHeight(
            190
        )

        right.addWidget(
            self.events_table,
            1,
        )

        right_scroll = QScrollArea()

        right_scroll.setObjectName(
            "rightScroll"
        )

        right_scroll.setWidgetResizable(
            True
        )

        right_scroll.setFrameShape(
            QFrame.NoFrame
        )

        right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        right_scroll.setWidget(
            right_panel
        )

        content.addWidget(
            camera_panel,
            3,
        )

        content.addWidget(
            right_scroll,
            2,
        )

        main.addLayout(
            content,
            1,
        )

        source_panel = (
            self.make_control_panel()
        )

        source = source_panel.layout()

        source.addWidget(
            self.control_label(
                "Источник:"
            )
        )

        source.addWidget(
            self.button(
                "Открыть видео",
                self.open_video,
            )
        )

        source.addWidget(
            self.button(
                "Веб-камера",
                self.open_webcam,
            )
        )

        source.addWidget(
            self.button(
                "Симулятор",
                self.open_simulator,
            )
        )

        source.addWidget(
            self.button(
                "Фото человека",
                self.load_person_sprite,
            )
        )

        source.addWidget(
            self.button(
                "Экспорт CSV",
                self.export_today_csv,
            )
        )

        source.addWidget(
            self.button(
                "Аналитика",
                self.show_analytics,
            )
        )

        source.addStretch()

        main.addWidget(
            source_panel
        )

        sim_panel = (
            self.make_control_panel()
        )

        sim = sim_panel.layout()

        sim.addWidget(
            self.control_label(
                "Симуляция:"
            )
        )

        sim.addWidget(
            self.button(
                "+ Входящий",
                self.add_entering_person,
            )
        )

        sim.addWidget(
            self.button(
                "+ Выходящий",
                self.add_exiting_person,
            )
        )

        sim.addWidget(
            self.button(
                "+ Группа",
                self.add_group,
            )
        )

        sim.addWidget(
            self.button(
                "Автотест 10 проходов",
                self.start_auto_test,
                "testButton",
            )
        )

        sim.addStretch()

        sim.addWidget(
            self.button(
                "Старт",
                self.start_processing,
                "primaryButton",
            )
        )

        sim.addWidget(
            self.button(
                "Пауза",
                self.pause_processing,
            )
        )

        sim.addWidget(
            self.button(
                "Очистить статистику",
                self.reset_statistics,
                "dangerButton",
            )
        )

        main.addWidget(
            sim_panel
        )

        self.apply_styles()

        self.update_statistics()

    def make_control_panel(self):

        panel = QFrame()

        panel.setObjectName(
            "panel"
        )

        layout = QHBoxLayout(
            panel
        )

        layout.setContentsMargins(
            14,
            9,
            14,
            9,
        )

        layout.setSpacing(8)

        return panel

    def button(
        self,
        text,
        slot,
        object_name="secondaryButton",
    ):

        button = QPushButton(text)

        button.setObjectName(
            object_name
        )

        button.clicked.connect(
            slot
        )

        return button

    def control_label(
        self,
        text,
    ):

        label = QLabel(text)

        label.setObjectName(
            "controlLabel"
        )

        return label

    def stat_card(
        self,
        title_text,
    ):

        card = QFrame()

        card.setObjectName(
            "statCard"
        )

        card.setMinimumHeight(
            96
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            10,
            8,
            10,
            8,
        )

        layout.setSpacing(2)

        title = QLabel(
            title_text
        )

        title.setObjectName(
            "statTitle"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        value = QLabel("0")

        value.setObjectName(
            "statValue"
        )

        value.setAlignment(
            Qt.AlignCenter
        )

        value.setMinimumHeight(
            44
        )

        layout.addWidget(title)
        layout.addWidget(value)

        return card, value

    @staticmethod
    def dialog_stylesheet():

        return """
            QDialog,
            QMessageBox {
                background: #111827;
            }

            QLabel {
                color: #f3f4f6;
                font-family: "Segoe UI";
                font-size: 14px;
            }

            QLabel#analyticsTitle {
                font-size: 22px;
                font-weight: 700;
            }

            QLabel#analyticsSummary {
                background: #0b1220;
                border: 1px solid #243148;
                border-radius: 10px;
                padding: 12px;
                color: #cbd5e1;
            }

            QTableWidget#analyticsTable {
                background: #0b1220;
                border: 1px solid #243148;
                border-radius: 10px;
                gridline-color: #1f2937;
                color: #f3f4f6;
            }

            QHeaderView::section {
                background: #172033;
                color: #cbd5e1;
                border: none;
                border-bottom: 1px solid #243148;
                padding: 8px;
                font-weight: 650;
            }

            QPushButton {
                color: white;
                background: #2563eb;
                border: none;
                border-radius: 7px;
                min-width: 90px;
                min-height: 34px;
                padding: 4px 12px;
                font-weight: 600;
            }

            QPushButton:hover {
                background: #1d4ed8;
            }
        """

    def show_message(
        self,
        title,
        text,
        icon=QMessageBox.Information,
    ):

        box = QMessageBox(self)

        box.setIcon(icon)

        box.setWindowTitle(
            title
        )

        box.setText(text)

        box.setStandardButtons(
            QMessageBox.Ok
        )

        box.setStyleSheet(
            self.dialog_stylesheet()
        )

        box.exec()

    def ask_yes_no(
        self,
        title,
        text,
    ):

        box = QMessageBox(self)

        box.setIcon(
            QMessageBox.Question
        )

        box.setWindowTitle(
            title
        )

        box.setText(text)

        box.setStandardButtons(
            QMessageBox.Yes
            |
            QMessageBox.No
        )

        box.setDefaultButton(
            QMessageBox.No
        )

        box.setStyleSheet(
            self.dialog_stylesheet()
        )

        return (
            box.exec()
            ==
            QMessageBox.Yes
        )

    def show_analytics(self):

        AnalyticsDialog(
            self,
            self.db,
        ).exec()

    def refresh_model_info(self):

        sprite = (
            self.person_sprite_name
            or
            "не загружено"
        )

        self.model_info.setText(
            f"Нейросеть: {MODEL_NAME}\n"
            "Класс: person\n"
            "Трекер: ByteTrack\n"
            f"Порог уверенности: "
            f"{CONFIDENCE:.2f}\n"
            f"Фото для симулятора: "
            f"{sprite}\n"
            f"База данных: "
            f"{self.db_path.name}"
        )

    def refresh_today_data(self):

        rows = (
            self.db.today_visits()
        )

        self.entered = sum(
            1
            for row in rows
            if row[2] == "ВХОД"
        )

        self.exited = sum(
            1
            for row in rows
            if row[2] == "ВЫХОД"
        )

        self.events_table.setRowCount(
            0
        )

        for (
            event_time,
            track_id,
            event_type,
            source,
        ) in rows:

            row_index = (
                self.events_table.rowCount()
            )

            self.events_table.insertRow(
                row_index
            )

            try:

                event_time = (
                    datetime
                    .fromisoformat(
                        event_time
                    )
                    .strftime(
                        "%H:%M:%S"
                    )
                )

            except ValueError:

                pass

            values = (
                event_time,
                track_id,
                event_type,
                source,
            )

            for column, value in enumerate(
                values
            ):

                self.events_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(
                        str(value)
                    ),
                )

        self.events_table.scrollToBottom()

        self.update_statistics()

        self.today_info.setText(
            f"Дата: "
            f"{datetime.now().strftime('%d.%m.%Y')}\n"

            f"Событий сегодня: "
            f"{len(rows)}\n"

            f"Входов: "
            f"{self.entered} | "

            f"Выходов: "
            f"{self.exited}"
        )

    def refresh_last_test(self):

        row = self.db.last_test()

        if not row:

            return

        (
            run_time,
            expected_in,
            expected_out,
            detected_in,
            detected_out,
            precision,
            recall,
            f1,
            accuracy,
        ) = row

        try:

            run_time = (
                datetime
                .fromisoformat(
                    run_time
                )
                .strftime(
                    "%d.%m.%Y %H:%M"
                )
            )

        except ValueError:

            pass

        self.test_info.setText(
            f"Последний автотест: "
            f"{run_time}\n"

            f"Эталон: вход "
            f"{expected_in}, "
            f"выход "
            f"{expected_out}\n"

            f"Система: вход "
            f"{detected_in}, "
            f"выход "
            f"{detected_out}\n"

            f"Precision: "
            f"{precision:.3f} | "

            f"Recall: "
            f"{recall:.3f}\n"

            f"F1-score: "
            f"{f1:.3f}\n"

            f"Точность подсчёта: "
            f"{accuracy * 100:.1f}%"
        )

    def export_today_csv(self):

        rows = self.db.today_visits()

        if not rows:

            self.show_message(
                "Экспорт",
                "За текущий день "
                "в журнале пока нет событий.",
            )

            return

        default_name = (
            f"visits_"
            f"{date.today().isoformat()}"
            f".csv"
        )

        path, _ = (
            QFileDialog.getSaveFileName(
                self,
                "Сохранить журнал посещений",
                str(
                    Path.home()
                    /
                    default_name
                ),
                "CSV (*.csv)",
            )
        )

        if not path:

            return

        if not path.lower().endswith(
            ".csv"
        ):

            path += ".csv"

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            writer = csv.writer(
                file,
                delimiter=";",
            )

            writer.writerow(
                [
                    "Дата и время",
                    "Track ID",
                    "Событие",
                    "Источник",
                ]
            )

            writer.writerows(rows)

        self.show_message(
            "Экспорт завершён",
            f"Журнал посещений сохранён:\n"
            f"{path}",
        )

    def load_model(self):

        try:

            self.model = YOLO(
                MODEL_NAME
            )

            self.set_status(
                "● YOLO11n загружена",
                True,
            )

        except Exception as error:

            self.model = None

            self.set_status(
                "● Ошибка загрузки YOLO11n",
                False,
            )

            self.show_message(
                "Ошибка нейросети",
                "Не удалось загрузить "
                "YOLO11n.\n\n"
                f"{error}",
                QMessageBox.Critical,
            )

    def reset_tracker(self):

        self.track_sides.clear()

        if self.model is not None:

            try:

                self.model.predictor = None

            except Exception:

                pass

    def open_video(self):

        if self.test_active:

            self.cancel_auto_test(
                True
            )

        path, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Выберите видео",
                "",
                "Видео "
                "(*.mp4 *.avi *.mov *.mkv);;"
                "Все файлы (*.*)",
            )
        )

        if not path:

            return

        self.close_video_source()

        capture = cv2.VideoCapture(
            path
        )

        if not capture.isOpened():

            self.show_message(
                "Ошибка",
                "Не удалось открыть "
                "выбранное видео.",
                QMessageBox.Warning,
            )

            return

        self.video_capture = capture

        self.source_mode = "video"

        self.source_badge.setText(
            "Видеофайл"
        )

        self.reset_tracker()

        self.start_processing()

    def open_webcam(self):

        if self.test_active:

            self.cancel_auto_test(
                True
            )

        self.close_video_source()

        capture = cv2.VideoCapture(
            0
        )

        if not capture.isOpened():

            self.show_message(
                "Ошибка",
                "Не удалось открыть "
                "веб-камеру.\n\n"

                "Проверьте, не используется ли "
                "она другой программой.",
                QMessageBox.Warning,
            )

            return

        self.video_capture = capture

        self.source_mode = "webcam"

        self.source_badge.setText(
            "Веб-камера"
        )

        self.reset_tracker()

        self.start_processing()

    def open_simulator(self):

        if self.test_active:

            self.cancel_auto_test(
                True
            )

        self.close_video_source()

        self.source_mode = (
            "simulation"
        )

        self.source_badge.setText(
            "Виртуальная проходная"
        )

        self.sim_people.clear()

        self.reset_tracker()

        frame = (
            self.render_simulation_frame(
                False
            )
        )

        frame = (
            self.draw_control_line_only(
                frame
            )
        )

        self.show_frame(frame)

        if self.person_sprite is not None:

            self.set_status(
                "● Симулятор готов",
                True,
            )

        else:

            self.set_status(
                "● Симулятор готов — "
                "загрузите фото человека",
                False,
            )

        self.timer.start(
            SIM_TIMER_MS
        )

    def load_person_sprite(self):

        path, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Выберите изображение человека",
                "",
                "Изображения "
                "(*.png *.jpg *.jpeg *.bmp)",
            )
        )

        if not path:

            return

        try:

            raw = np.fromfile(
                path,
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                raw,
                cv2.IMREAD_UNCHANGED,
            )

        except Exception:

            image = None

        if image is None:

            self.show_message(
                "Ошибка",
                "Не удалось открыть изображение.",
                QMessageBox.Warning,
            )

            return

        self.person_sprite = image

        self.person_sprite_name = (
            Path(path).name
        )

        self.refresh_model_info()

        if self.source_mode != "simulation":

            self.open_simulator()

        else:

            self.set_status(
                "● Фото человека загружено",
                True,
            )

    def ensure_simulator_ready(self):

        if self.source_mode != "simulation":

            self.open_simulator()

        if self.person_sprite is None:

            self.show_message(
                "Фото человека",

                "Сначала загрузите "
                "фотографию человека "
                "в полный рост.\n\n"

                "Лучше использовать "
                "изображение с одним человеком "
                "и простым фоном.",
            )

            self.load_person_sprite()

        return (
            self.person_sprite
            is not None
        )

    def add_entering_person(self):

        if self.test_active:

            self.show_message(
                "Автотест",
                "Дождитесь завершения автотеста.",
            )

            return

        if self.ensure_simulator_ready():

            self.spawn_person(1)

            self.timer.start(
                SIM_TIMER_MS
            )

    def add_exiting_person(self):

        if self.test_active:

            self.show_message(
                "Автотест",
                "Дождитесь завершения автотеста.",
            )

            return

        if self.ensure_simulator_ready():

            self.spawn_person(-1)

            self.timer.start(
                SIM_TIMER_MS
            )

    def add_group(self):

        if self.test_active:

            self.show_message(
                "Автотест",
                "Дождитесь завершения автотеста.",
            )

            return

        if not self.ensure_simulator_ready():

            return

        for index, x in enumerate(
            (
                240,
                480,
                720,
            )
        ):

            self.spawn_person(
                1,
                x=x,
                delay=index * 20,
            )

        self.timer.start(
            SIM_TIMER_MS
        )

    def spawn_person(
        self,
        direction,
        x=None,
        delay=0,
        speed=None,
    ):

        if x is None:

            x = random.randint(
                140,
                SIM_WIDTH - 160,
            )

        if direction == 1:

            y = (
                -SIM_PERSON_HEIGHT
                -
                delay
            )

        else:

            y = (
                SIM_HEIGHT
                +
                delay
            )

        if speed is None:

            speed = random.uniform(
                MANUAL_SPEED_MIN,
                MANUAL_SPEED_MAX,
            )

        self.sim_people.append(
            SimPerson(
                float(x),
                float(y),
                int(direction),
                float(speed),
            )
        )

    def render_simulation_frame(
        self,
        move_people=True,
    ):

        frame = np.zeros(
            (
                SIM_HEIGHT,
                SIM_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        frame[
            :260,
            :
        ] = (
            63,
            72,
            84,
        )

        frame[
            260:,
            :
        ] = (
            31,
            43,
            56,
        )

        cv2.rectangle(
            frame,
            (
                0,
                235,
            ),
            (
                SIM_WIDTH,
                265,
            ),
            (
                89,
                99,
                110,
            ),
            -1,
        )

        for x in range(
            30,
            SIM_WIDTH,
            100,
        ):

            cv2.rectangle(
                frame,
                (
                    x,
                    248,
                ),
                (
                    x + 48,
                    253,
                ),
                (
                    205,
                    210,
                    215,
                ),
                -1,
            )

        line_y = int(
            SIM_HEIGHT
            *
            LINE_POSITION
        )

        cv2.rectangle(
            frame,
            (
                0,
                line_y - 90,
            ),
            (
                245,
                line_y + 90,
            ),
            (
                95,
                102,
                112,
            ),
            -1,
        )

        cv2.rectangle(
            frame,
            (
                715,
                line_y - 90,
            ),
            (
                SIM_WIDTH,
                line_y + 90,
            ),
            (
                95,
                102,
                112,
            ),
            -1,
        )

        cv2.rectangle(
            frame,
            (
                245,
                line_y - 90,
            ),
            (
                715,
                line_y + 90,
            ),
            (
                48,
                61,
                75,
            ),
            -1,
        )

        cv2.rectangle(
            frame,
            (
                245,
                line_y - 90,
            ),
            (
                715,
                line_y + 90,
            ),
            (
                126,
                138,
                150,
            ),
            2,
        )

        for x in (
            350,
            480,
            610,
        ):

            cv2.rectangle(
                frame,
                (
                    x - 18,
                    line_y - 35,
                ),
                (
                    x + 18,
                    line_y + 35,
                ),
                (
                    120,
                    132,
                    145,
                ),
                -1,
            )

            cv2.line(
                frame,
                (
                    x,
                    line_y,
                ),
                (
                    x + 68,
                    line_y - 25,
                ),
                (
                    220,
                    225,
                    230,
                ),
                5,
            )

        cv2.putText(
            frame,
            "OUTSIDE",
            (
                28,
                42,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (
                230,
                235,
                240,
            ),
            2,
        )

        cv2.putText(
            frame,
            "ENTERPRISE",
            (
                28,
                SIM_HEIGHT - 26,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (
                230,
                235,
                240,
            ),
            2,
        )

        cv2.putText(
            frame,
            "VIRTUAL ENTRANCE",
            (
                350,
                34,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                185,
                205,
                230,
            ),
            2,
        )

        if self.person_sprite is not None:

            for person in list(
                self.sim_people
            ):

                self.overlay_person(
                    frame,
                    person,
                )

                if move_people:

                    person.y += (
                        person.speed
                        *
                        person.direction
                    )

                if (
                    person.direction == 1
                    and
                    person.y
                    >
                    SIM_HEIGHT + 40
                ):

                    self.sim_people.remove(
                        person
                    )

                elif (
                    person.direction == -1
                    and
                    person.y
                    <
                    -person.height - 40
                ):

                    self.sim_people.remove(
                        person
                    )

        return frame

    def overlay_person(
        self,
        frame,
        person: SimPerson,
    ):

        sprite = self.person_sprite

        if sprite is None:

            return

        original_height, original_width = (
            sprite.shape[:2]
        )

        target_height = int(
            person.height
        )

        target_width = max(
            1,
            int(
                original_width
                *
                target_height
                /
                max(
                    original_height,
                    1,
                )
            ),
        )

        resized = cv2.resize(
            sprite,
            (
                target_width,
                target_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        x = int(
            person.x
            -
            target_width / 2
        )

        y = int(
            person.y
        )

        x1 = max(
            0,
            x,
        )

        y1 = max(
            0,
            y,
        )

        x2 = min(
            frame.shape[1],
            x + target_width,
        )

        y2 = min(
            frame.shape[0],
            y + target_height,
        )

        if (
            x1 >= x2
            or
            y1 >= y2
        ):

            return

        sx1 = x1 - x
        sy1 = y1 - y

        crop = resized[
            sy1:
            sy1 + (y2 - y1),

            sx1:
            sx1 + (x2 - x1)
        ]

        if (
            crop.ndim == 3
            and
            crop.shape[2] == 4
        ):

            rgb = (
                crop[
                    :,
                    :,
                    :3
                ]
                .astype(
                    np.float32
                )
            )

            alpha = (
                crop[
                    :,
                    :,
                    3:4
                ]
                .astype(
                    np.float32
                )
                /
                255.0
            )

            background = (
                frame[
                    y1:y2,
                    x1:x2
                ]
                .astype(
                    np.float32
                )
            )

            frame[
                y1:y2,
                x1:x2
            ] = (
                rgb * alpha
                +
                background
                *
                (
                    1.0 - alpha
                )
            ).astype(
                np.uint8
            )

        else:

            frame[
                y1:y2,
                x1:x2
            ] = (
                crop[
                    :,
                    :,
                    :3
                ]
            )

    def start_auto_test(self):

        if self.test_active:

            self.show_message(
                "Автотест",
                "Автотест уже выполняется.",
            )

            return

        if not self.ensure_simulator_ready():

            return

        self.source_mode = (
            "simulation"
        )

        self.source_badge.setText(
            "Автотест"
        )

        self.sim_people.clear()

        self.reset_tracker()

        self.test_active = True

        self.test_plan = list(
            AUTO_TEST_PLAN
        )

        self.test_expected_in = sum(
            1
            for item in self.test_plan
            if item == 1
        )

        self.test_expected_out = sum(
            1
            for item in self.test_plan
            if item == -1
        )

        self.test_detected_in = 0
        self.test_detected_out = 0
        self.test_idle_frames = 0

        self.test_info.setText(
            "Автотест выполняется...\n"

            f"Эталон: вход "
            f"{self.test_expected_in}, "
            f"выход "
            f"{self.test_expected_out}\n"

            "Система: вход 0, выход 0\n"

            "Precision / Recall / F1 "
            "будут рассчитаны "
            "после завершения."
        )

        self.spawn_next_test_person()

        self.timer.start(
            SIM_TIMER_MS
        )

        self.set_status(
            "● Выполняется "
            "автоматизированный тест",
            True,
        )

    def spawn_next_test_person(self):

        if not self.test_plan:

            return

        direction = (
            self.test_plan.pop(0)
        )

        speed = random.uniform(
            AUTO_TEST_SPEED_MIN,
            AUTO_TEST_SPEED_MAX,
        )

        x = random.choice(
            (
                330,
                480,
                630,
            )
        )

        self.spawn_person(
            direction,
            x=x,
            speed=speed,
        )

    def update_auto_test(self):

        if not self.test_active:

            return

        if self.sim_people:

            self.test_idle_frames = 0

            return

        self.test_idle_frames += 1

        if self.test_plan:

            if (
                self.test_idle_frames
                >=
                AUTO_TEST_GAP_FRAMES
            ):

                self.test_idle_frames = 0

                self.reset_tracker()

                self.spawn_next_test_person()

            return

        if (
            self.test_idle_frames
            >=
            AUTO_TEST_FINISH_FRAMES
        ):

            self.finish_auto_test()

    @staticmethod
    def calculate_metrics(
        expected_in,
        expected_out,
        detected_in,
        detected_out,
    ):

        true_positive = (
            min(
                expected_in,
                detected_in,
            )
            +
            min(
                expected_out,
                detected_out,
            )
        )

        false_positive = (
            max(
                detected_in
                -
                expected_in,
                0,
            )
            +
            max(
                detected_out
                -
                expected_out,
                0,
            )
        )

        false_negative = (
            max(
                expected_in
                -
                detected_in,
                0,
            )
            +
            max(
                expected_out
                -
                detected_out,
                0,
            )
        )

        if (
            true_positive
            +
            false_positive
        ):

            precision = (
                true_positive
                /
                (
                    true_positive
                    +
                    false_positive
                )
            )

        else:

            precision = 1.0

        if (
            true_positive
            +
            false_negative
        ):

            recall = (
                true_positive
                /
                (
                    true_positive
                    +
                    false_negative
                )
            )

        else:

            recall = 1.0

        if (
            precision
            +
            recall
        ):

            f1 = (
                2
                *
                precision
                *
                recall
                /
                (
                    precision
                    +
                    recall
                )
            )

        else:

            f1 = 0.0

        total = (
            expected_in
            +
            expected_out
        )

        error = (
            abs(
                detected_in
                -
                expected_in
            )
            +
            abs(
                detected_out
                -
                expected_out
            )
        )

        if total:

            accuracy = max(
                0.0,
                1.0
                -
                error / total,
            )

        else:

            accuracy = 1.0

        return (
            precision,
            recall,
            f1,
            accuracy,
        )

    def finish_auto_test(self):

        expected_in = (
            self.test_expected_in
        )

        expected_out = (
            self.test_expected_out
        )

        detected_in = (
            self.test_detected_in
        )

        detected_out = (
            self.test_detected_out
        )

        (
            precision,
            recall,
            f1,
            accuracy,
        ) = self.calculate_metrics(
            expected_in,
            expected_out,
            detected_in,
            detected_out,
        )

        self.db.save_test(
            expected_in,
            expected_out,
            detected_in,
            detected_out,
            precision,
            recall,
            f1,
            accuracy,
        )

        self.test_active = False

        self.timer.stop()

        self.source_badge.setText(
            "Виртуальная проходная"
        )

        self.test_info.setText(
            "Автотест завершён.\n"

            f"Эталон: вход "
            f"{expected_in}, "
            f"выход "
            f"{expected_out}\n"

            f"Система: вход "
            f"{detected_in}, "
            f"выход "
            f"{detected_out}\n"

            f"Precision: "
            f"{precision:.3f} | "

            f"Recall: "
            f"{recall:.3f}\n"

            f"F1-score: "
            f"{f1:.3f}\n"

            f"Точность подсчёта: "
            f"{accuracy * 100:.1f}%"
        )

        self.set_status(
            "● Автотест завершён",
            True,
        )

        self.show_message(
            "Результаты автотеста",

            f"Ожидалось:\n"
            f"{expected_in} входов и "
            f"{expected_out} выходов\n\n"

            f"Распознано:\n"
            f"{detected_in} входов и "
            f"{detected_out} выходов\n\n"

            f"Precision: "
            f"{precision:.3f}\n"

            f"Recall: "
            f"{recall:.3f}\n"

            f"F1-score: "
            f"{f1:.3f}\n"

            f"Точность подсчёта: "
            f"{accuracy * 100:.1f}%"
        )

    def cancel_auto_test(
        self,
        silent=False,
    ):

        if not self.test_active:

            return

        self.test_active = False

        self.test_plan.clear()

        self.sim_people.clear()

        self.test_idle_frames = 0

        if not silent:

            self.set_status(
                "● Автотест отменён",
                False,
            )

    def start_processing(self):

        if self.model is None:

            self.show_message(
                "YOLO11n",
                "Нейросетевая модель "
                "не загружена.",
            )

            return

        if self.source_mode == "simulation":

            self.timer.start(
                SIM_TIMER_MS
            )

            self.set_status(
                "● Виртуальная проходная работает",
                True,
            )

            return

        if (
            self.video_capture is None
            or
            not self.video_capture.isOpened()
        ):

            self.show_message(
                "Источник",
                "Сначала выберите "
                "источник видеопотока.",
            )

            return

        self.timer.start(
            VIDEO_TIMER_MS
        )

        self.set_status(
            "● Анализ видеопотока выполняется",
            True,
        )

    def pause_processing(self):

        if self.test_active:

            self.show_message(
                "Автотест",
                "Во время автотеста "
                "пауза отключена.",
            )

            return

        self.timer.stop()

        self.set_status(
            "● Анализ приостановлен",
            False,
        )

    def process_next_frame(self):

        if self.model is None:

            return

        if self.source_mode == "simulation":

            frame = (
                self.render_simulation_frame(
                    True
                )
            )

        else:

            if self.video_capture is None:

                return

            success, frame = (
                self.video_capture.read()
            )

            if not success:

                self.timer.stop()

                self.set_status(
                    "● Видео завершено",
                    False,
                )

                return

        analyzed = self.analyze_frame(
            frame
        )

        self.show_frame(
            analyzed
        )

        if self.source_mode == "simulation":

            self.update_auto_test()

    def analyze_frame(
        self,
        frame,
    ):

        height, width = (
            frame.shape[:2]
        )

        line_y = int(
            height
            *
            LINE_POSITION
        )

        results = self.model.track(
            frame,

            persist=True,

            tracker="bytetrack.yaml",

            classes=[
                PERSON_CLASS_ID
            ],

            conf=CONFIDENCE,

            verbose=False,
        )

        boxes = (
            results[0].boxes
        )

        self.draw_control_line(
            frame,
            line_y,
            width,
        )

        if (
            boxes is None
            or
            len(boxes) == 0
            or
            boxes.id is None
        ):

            return frame

        coordinates = (
            boxes
            .xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            boxes
            .id
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            boxes
            .conf
            .cpu()
            .numpy()
        )

        for (
            box,
            track_id,
            confidence,
        ) in zip(
            coordinates,
            track_ids,
            confidences,
        ):

            x1, y1, x2, y2 = map(
                int,
                box,
            )

            center_x = int(
                (
                    x1 + x2
                )
                /
                2
            )

            center_y = int(
                (
                    y1 + y2
                )
                /
                2
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                (
                    56,
                    189,
                    248,
                ),
                2,
            )

            cv2.circle(
                frame,
                (
                    center_x,
                    center_y,
                ),
                5,
                (
                    255,
                    255,
                    255,
                ),
                -1,
            )

            cv2.putText(
                frame,

                f"Person #{track_id}  "
                f"{confidence:.0%}",

                (
                    x1,
                    max(
                        25,
                        y1 - 10,
                    ),
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (
                    56,
                    189,
                    248,
                ),

                2,

                cv2.LINE_AA,
            )

            self.check_line_crossing(
                track_id,
                center_y,
                line_y,
            )

        return frame

    def draw_control_line(
        self,
        frame,
        line_y,
        width,
    ):

        cv2.line(
            frame,
            (
                0,
                line_y,
            ),
            (
                width,
                line_y,
            ),
            (
                0,
                215,
                255,
            ),
            3,
        )

        cv2.putText(
            frame,

            "ENTRY / EXIT LINE",

            (
                18,
                max(
                    30,
                    line_y - 12,
                ),
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (
                0,
                215,
                255,
            ),

            2,

            cv2.LINE_AA,
        )

    def draw_control_line_only(
        self,
        frame,
    ):

        height, width = (
            frame.shape[:2]
        )

        self.draw_control_line(
            frame,
            int(
                height
                *
                LINE_POSITION
            ),
            width,
        )

        return frame

    def check_line_crossing(
        self,
        track_id,
        center_y,
        line_y,
    ):

        if (
            center_y
            <
            line_y - LINE_MARGIN
        ):

            side = -1

        elif (
            center_y
            >
            line_y + LINE_MARGIN
        ):

            side = 1

        else:

            return

        old_side = (
            self.track_sides.get(
                track_id
            )
        )

        if (
            old_side == -1
            and
            side == 1
        ):

            self.add_event(
                track_id,
                "ВХОД",
            )

        elif (
            old_side == 1
            and
            side == -1
        ):

            self.add_event(
                track_id,
                "ВЫХОД",
            )

        self.track_sides[
            track_id
        ] = side

    def source_name(self):

        if self.test_active:

            return "Автотест"

        names = {
            "video": "Видеофайл",
            "webcam": "Веб-камера",
            "simulation": "Симулятор",
        }

        return names.get(
            self.source_mode,
            "Неизвестно",
        )

    def add_event(
        self,
        track_id,
        event_type,
    ):

        self.db.add_visit(
            track_id,
            event_type,
            self.source_name(),
        )

        if self.test_active:

            if event_type == "ВХОД":

                self.test_detected_in += 1

            elif event_type == "ВЫХОД":

                self.test_detected_out += 1

            remaining = (
                len(self.test_plan)
                +
                len(self.sim_people)
            )

            self.test_info.setText(
                "Автотест выполняется...\n"

                f"Эталон: вход "
                f"{self.test_expected_in}, "
                f"выход "
                f"{self.test_expected_out}\n"

                f"Система: вход "
                f"{self.test_detected_in}, "
                f"выход "
                f"{self.test_detected_out}\n"

                f"Осталось сценариев: "
                f"{remaining}"
            )

        self.refresh_today_data()

    def update_statistics(self):

        self.entered_label.setText(
            str(self.entered)
        )

        self.exited_label.setText(
            str(self.exited)
        )

        inside = max(
            self.entered
            -
            self.exited,
            0,
        )

        self.inside_label.setText(
            str(inside)
        )

    def reset_statistics(self):

        if self.test_active:

            self.show_message(
                "Автотест",
                "Сначала дождитесь "
                "завершения автотеста.",
            )

            return

        answer = self.ask_yes_no(
            "Очистить статистику",

            "Удалить все зарегистрированные "
            "события за текущий день "
            "из локальной базы данных?",
        )

        if not answer:

            return

        self.db.clear_today_visits()

        self.track_sides.clear()

        self.refresh_today_data()

    def show_frame(
        self,
        frame,
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        height, width, channels = (
            rgb.shape
        )

        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()

        pixmap = (
            QPixmap
            .fromImage(image)
            .scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        self.camera_label.setPixmap(
            pixmap
        )

    def set_status(
        self,
        text,
        running,
    ):

        self.status_label.setText(
            text
        )

        self.status_label.setObjectName(
            "statusRunning"
            if running
            else
            "statusStopped"
        )

        self.status_label.style().unpolish(
            self.status_label
        )

        self.status_label.style().polish(
            self.status_label
        )

    def close_video_source(self):

        self.timer.stop()

        if self.video_capture is not None:

            self.video_capture.release()

        self.video_capture = None

    def closeEvent(
        self,
        event,
    ):

        self.close_video_source()

        event.accept()

    def apply_styles(self):

        self.setStyleSheet(
            """
            QMainWindow {
                background: #0b1220;
            }

            QWidget {
                color: #f3f4f6;
                font-family: "Segoe UI";
                font-size: 14px;
            }

            QLabel#title {
                font-size: 25px;
                font-weight: 700;
            }

            QLabel#subtitle {
                color: #94a3b8;
                font-size: 13px;
            }

            QLabel#sectionTitle {
                font-size: 17px;
                font-weight: 650;
            }

            QLabel#controlLabel {
                color: #94a3b8;
                font-size: 13px;
                font-weight: 650;
            }

            QLabel#badge {
                background: #1d4ed8;
                color: white;
                border-radius: 8px;
                padding: 4px 9px;
                font-size: 11px;
                font-weight: 700;
            }

            QLabel#statusRunning {
                color: #22c55e;
                font-weight: 650;
            }

            QLabel#statusStopped {
                color: #f59e0b;
                font-weight: 650;
            }

            QFrame#panel {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 14px;
            }

            QLabel#camera {
                background: #050a13;
                border: 1px solid #253047;
                border-radius: 12px;
                color: #94a3b8;
                font-size: 16px;
                padding: 20px;
            }

            QLabel#hint {
                color: #64748b;
                font-size: 12px;
                padding: 4px;
            }

            QLabel#infoCard {
                background: #0b1220;
                border: 1px solid #243148;
                border-radius: 10px;
                color: #cbd5e1;
                padding: 9px;
                font-size: 12px;
            }

            QFrame#statCard {
                background: #0b1220;
                border: 1px solid #243148;
                border-radius: 12px;
            }

            QLabel#statTitle {
                color: #94a3b8;
                font-size: 12px;
                font-weight: 600;
            }

            QLabel#statValue {
                color: #f8fafc;
                font-size: 34px;
                font-weight: 750;
            }

            QTableWidget#eventsTable {
                background: #0b1220;
                border: 1px solid #243148;
                border-radius: 10px;
                gridline-color: #1f2937;
                selection-background-color: #1d4ed8;
            }

            QTableWidget#eventsTable::item {
                padding: 7px;
            }

            QHeaderView::section {
                background: #172033;
                color: #cbd5e1;
                border: none;
                border-bottom: 1px solid #243148;
                padding: 8px;
                font-weight: 650;
            }

            QScrollArea#rightScroll {
                border: none;
                background: transparent;
            }

            QScrollBar:vertical {
                background: #0b1220;
                width: 10px;
                margin: 3px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: #334155;
                min-height: 28px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background: #475569;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QPushButton {
                min-height: 38px;
                border: none;
                border-radius: 9px;
                padding: 0 16px;
                font-weight: 650;
            }

            QPushButton#primaryButton {
                background: #2563eb;
                color: white;
            }

            QPushButton#primaryButton:hover {
                background: #1d4ed8;
            }

            QPushButton#secondaryButton {
                background: #1f2937;
                color: #e5e7eb;
                border: 1px solid #334155;
            }

            QPushButton#secondaryButton:hover {
                background: #293548;
            }

            QPushButton#testButton {
                background: #0f766e;
                color: white;
                border: 1px solid #14b8a6;
            }

            QPushButton#testButton:hover {
                background: #0d9488;
            }

            QPushButton#dangerButton {
                background: #3f1d24;
                color: #fecaca;
                border: 1px solid #7f1d1d;
            }

            QPushButton#dangerButton:hover {
                background: #5a2029;
            }
            """
        )


def main():

    app = QApplication(
        sys.argv
    )

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":

    main()

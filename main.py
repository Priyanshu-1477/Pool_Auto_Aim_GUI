import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QSlider,
                             QPushButton, QHBoxLayout, QCheckBox, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent

BASE_WIDTH, BASE_HEIGHT = 800, 400
BALL_RADIUS = 10
POCKET_RADIUS = 15
MAX_BALLS = 5
ANGLE_THRESHOLD = 120
MAX_SHOTS = 3

pockets = [
    (0, 0), (BASE_WIDTH // 2, 0), (BASE_WIDTH, 0),
    (0, BASE_HEIGHT), (BASE_WIDTH // 2, BASE_HEIGHT), (BASE_WIDTH, BASE_HEIGHT)
]

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def calculate_angle(cue, ghost, target):
    v1 = np.array(ghost) - np.array(cue)
    v2 = np.array(target) - np.array(ghost)
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return 180 - np.degrees(np.arccos(dot))

def get_ghost_ball(target, pocket):
    direction = np.array(pocket) - np.array(target)
    direction = direction / np.linalg.norm(direction)
    return (np.array(target) - direction * BALL_RADIUS * 2).astype(int)

def is_path_clear(start, end, obstacles):
    for ball in obstacles:
        if np.allclose(start, ball) or np.allclose(end, ball): continue
        dist = np.abs(np.cross(end - start, start - ball)) / np.linalg.norm(end - start)
        proj = np.dot(ball - start, end - start) / np.linalg.norm(end - start)
        if 0 < proj < np.linalg.norm(end - start) and dist < BALL_RADIUS * 2:
            return False
    return True

def find_all_direct_shots(cue, targets):
    shots = []
    for target in targets:
        for pocket in pockets:
            ghost = get_ghost_ball(target, pocket)
            angle = calculate_angle(cue, ghost, target)
            if angle < ANGLE_THRESHOLD: continue
            if not is_path_clear(np.array(cue), ghost, targets): continue
            if not is_path_clear(np.array(target), np.array(pocket), targets): continue
            score = distance(cue, ghost) + distance(target, pocket) + abs(180 - angle) * 3
            shots.append({"cue": cue, "ghost": ghost, "target": target, "pocket": pocket, "angle": angle, "score": score})
    return sorted(shots, key=lambda s: s['score'])[:MAX_SHOTS]

def auto_place_cue(targets):
    best = None
    for x in range(50, BASE_WIDTH - 50, 40):
        for y in range(50, BASE_HEIGHT - 50, 40):
            cue = [x, y]
            shots = find_all_direct_shots(cue, [np.array(b) for b in targets])
            if shots:
                if not best or shots[0]['score'] < best['shot']['score']:
                    best = {"cue": cue, "shot": shots[0]}
    return best

def draw_table(cue, targets, shot=None, shot_index=0, scale=1.0):
    w, h = int(BASE_WIDTH * scale), int(BASE_HEIGHT * scale)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (0, 128, 0)
    scaled_pockets = [(int(p[0] * scale), int(p[1] * scale)) for p in pockets]

    for p in scaled_pockets:
        cv2.circle(img, p, int(POCKET_RADIUS * scale), (0, 0, 0), -1)

    cv2.circle(img, tuple((np.array(cue) * scale).astype(int)), int(BALL_RADIUS * scale), (255, 255, 255), -1)
    for ball in targets:
        cv2.circle(img, tuple((np.array(ball) * scale).astype(int)), int(BALL_RADIUS * scale), (0, 0, 255), -1)

    if shot:
        cue_scaled = (np.array(shot['cue']) * scale).astype(int)
        ghost_scaled = (np.array(shot['ghost']) * scale).astype(int)
        target_scaled = (np.array(shot['target']) * scale).astype(int)
        pocket_scaled = (np.array(shot['pocket']) * scale).astype(int)

        cv2.circle(img, tuple(ghost_scaled), int(BALL_RADIUS * scale), (0, 255, 255), 2)
        cv2.line(img, tuple(cue_scaled), tuple(ghost_scaled), (255, 255, 0), 2)
        cv2.line(img, tuple(target_scaled), tuple(pocket_scaled), (255, 0, 255), 2)
        trail_dir = ghost_scaled - cue_scaled
        trail_dir = trail_dir / np.linalg.norm(trail_dir)
        trail_end = (ghost_scaled + (trail_dir * 50)).astype(int)
        cv2.line(img, tuple(ghost_scaled), tuple(trail_end), (100, 255, 255), 1)
        mid = ((cue_scaled + ghost_scaled) // 2).astype(int)
        cv2.putText(img, f"{shot['angle']:.1f} deg", tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img

class PoolApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("8-Ball Auto Aim (Resizable)")
        self.cue = [300, 200]
        self.targets = [[400, 200], [420, 200], [380, 200], [460, 180], [470, 220]]
        self.active = [True] * MAX_BALLS
        self.shots = []
        self.shot_index = 0
        self.scale = 1.0

        self.canvas = QLabel()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.sliders = []
        grid = QGridLayout()
        for i in range(MAX_BALLS):
            sx = QSlider(Qt.Horizontal); sx.setMaximum(BASE_WIDTH); sx.setValue(self.targets[i][0])
            sy = QSlider(Qt.Horizontal); sy.setMaximum(BASE_HEIGHT); sy.setValue(self.targets[i][1])
            cb = QCheckBox("Ball {}".format(i+1)); cb.setChecked(True)
            sx.valueChanged.connect(self.update_scene)
            sy.valueChanged.connect(self.update_scene)
            cb.stateChanged.connect(self.update_scene)
            self.sliders.append((sx, sy, cb))
            grid.addWidget(cb, i, 0)
            grid.addWidget(sx, i, 1)
            grid.addWidget(sy, i, 2)

        self.cue_x = QSlider(Qt.Horizontal); self.cue_x.setMaximum(BASE_WIDTH); self.cue_x.setValue(self.cue[0])
        self.cue_y = QSlider(Qt.Horizontal); self.cue_y.setMaximum(BASE_HEIGHT); self.cue_y.setValue(self.cue[1])
        self.cue_x.valueChanged.connect(self.update_scene)
        self.cue_y.valueChanged.connect(self.update_scene)

        self.btn_auto = QPushButton("Auto Cue")
        self.btn_next = QPushButton("Next Shot")
        self.btn_prev = QPushButton("Prev Shot")
        self.btn_auto.clicked.connect(self.auto_cue)
        self.btn_next.clicked.connect(self.next_shot)
        self.btn_prev.clicked.connect(self.prev_shot)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Cue Position"))
        layout.addWidget(self.cue_x)
        layout.addWidget(self.cue_y)
        layout.addLayout(grid)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_auto)
        btns.addWidget(self.btn_prev)
        btns.addWidget(self.btn_next)
        layout.addLayout(btns)

        self.setLayout(layout)
        self.resize(BASE_WIDTH + 300, BASE_HEIGHT + 200)
        self.update_scene()

    def resizeEvent(self, event: QResizeEvent):
        w = self.canvas.width()
        self.scale = w / BASE_WIDTH
        self.update_scene()

    def get_active_targets(self):
        updated = []
        for i, (sx, sy, cb) in enumerate(self.sliders):
            if cb.isChecked():
                updated.append([sx.value(), sy.value()])
        return updated

    def update_scene(self):
        self.cue = [self.cue_x.value(), self.cue_y.value()]
        self.targets = self.get_active_targets()
        self.shots = find_all_direct_shots(self.cue, [np.array(b) for b in self.targets])
        self.shot_index = 0
        shot = self.shots[self.shot_index] if self.shots else None
        img = draw_table(self.cue, self.targets, shot, self.shot_index, self.scale)
        self.display(img)

    def auto_cue(self):
        result = auto_place_cue([np.array(b) for b in self.get_active_targets()])
        if result:
            self.cue = result['cue']
            self.cue_x.setValue(self.cue[0])
            self.cue_y.setValue(self.cue[1])
        self.update_scene()

    def next_shot(self):
        if self.shots:
            self.shot_index = (self.shot_index + 1) % len(self.shots)
            img = draw_table(self.cue, self.targets, self.shots[self.shot_index], self.shot_index, self.scale)
            self.display(img)

    def prev_shot(self):
        if self.shots:
            self.shot_index = (self.shot_index - 1) % len(self.shots)
            img = draw_table(self.cue, self.targets, self.shots[self.shot_index], self.shot_index, self.scale)
            self.display(img)

    def display(self, img):
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        self.canvas.setPixmap(QPixmap.fromImage(qimg).scaled(self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = PoolApp()
    win.show()
    sys.exit(app.exec_())

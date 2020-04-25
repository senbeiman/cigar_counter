import json
import time

import cv2
import numpy as np
import requests

BINARY_THRESHOLD = 100
CIGAR_NUM = 10
LOG_LENGTH = 5
POST_URL = json.load(open('url.json', 'r'))["post_url"]


def main():
    # from pi camera
    cap = cv2.VideoCapture(0)
    count_controller = CountController()
    timer = cv2.TickMeter()  # FPS測定用
    timer.start()
    while cap.isOpened():
        ret, frame = cap.read()

        # ARマーカーの検出
        frame_color = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        left_marker, right_marker = detect_markers(frame_color)

        # タバコのカウント
        ret, count = count_cigars(frame_color, left_marker, right_marker)
        if not ret:
            continue
        count_controller.log_count(count)

        if cv2.waitKey(1) != -1:
            break
        time.sleep(max([0, 1-timer.getTimeSec()]))  # FPSを1にするための調整
        timer.reset()
        timer.start()
    cap.release()
    cv2.destroyAllWindows()


class CountController:
    def __init__(self):
        self.count_log = [-1] * LOG_LENGTH
        self.count = 0

    def log_count(self, count):
        self.count_log = self.count_log[1:] + [count]
        if len(set(self.count_log)) == 1:
            if self.count - count == 1:
                requests.post(POST_URL,
                              json.dumps({'timestamp': str(int(time.time()))}),
                              headers={'Content-Type': 'application/json'})
                print('timestamp has posted:', str(int(time.time())))
            self.count = count
            print(self.count)


def detect_markers(frame):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
    left_marker = None
    right_marker = None
    if ids is not None:
        for i, c in zip(ids, corners):
            marker = [tuple(x) for i, x in enumerate(c[0].astype(np.int)) if i%2 == 0]
            if i[0] == 1:
                right_marker = marker
            elif i[0] == 2:
                left_marker = marker
    if left_marker is not None:
        cv2.rectangle(frame, left_marker[0], left_marker[1], (0, 255, 0), 1)
    if right_marker is not None:
        cv2.rectangle(frame, right_marker[0], right_marker[1], (0, 255, 0), 1)
    cv2.imshow('detection', frame)
    return left_marker, right_marker


def count_cigars(frame_color, left_marker, right_marker):
    if left_marker is None or right_marker is None:
        return False, None
    lx1, ly1 = left_marker[0]
    lx2, ly2 = left_marker[1]
    rx2, ry2 = right_marker[1]
    marker_height = abs(ly2 - ly1)
    y = int(marker_height * 2  / 3)
    count = 0
    frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    _, frame_binary = cv2.threshold(frame_gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    frame_binary_roi = frame_binary[ly1-marker_height: ly1, lx1: rx2+1]
    frame_binary_color_roi = cv2.cvtColor(frame_binary_roi, cv2.COLOR_GRAY2BGR)
    for i in range(CIGAR_NUM):
        x = int((rx2 - lx1)/127 * (5 + 13 * i))
        cv2.drawMarker(frame_binary_color_roi, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=5)
        if frame_binary_roi[y, x] == 255:
            count += 1
    cv2.putText(frame_binary_color_roi, "count: {:}".format(count), (0, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 255, 0), thickness=1)
    cv2.imshow('roi', frame_binary_color_roi)
    return True, count


if __name__ == '__main__':
    main()

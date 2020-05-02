import os
import collections
import datetime
import json
import time

import cv2
import numpy as np
import requests

Marker = collections.namedtuple('Marker', ['id', 'left_top', 'right_bottom'])
PROJECT_DIR = os.path.dirname(__file__)
JSON_PATH = os.path.join(PROJECT_DIR, 'url.json')


class SlackPoster:
    def __init__(self):
        self.url = "https://slack.com/api/chat.postMessage"
        self.token = json.load(open(JSON_PATH, 'r'))["slack_token"]
        self.channel = json.load(open(JSON_PATH, 'r'))["slack_channel"]

    def post_message(self, message):
        data = {
            "token": self.token,
            "channel": self.channel,
            "text": message
        }
        requests.post(self.url, data=data)
        print("message posted to slack")


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = cv2.TickMeter()  # FPS調整用
        self.timer.start()

    def read_frame(self):
        time.sleep(max([0, 1-self.timer.getTimeSec()]))  # FPSを1にするための調整
        self.timer.reset()
        self.timer.start()
        ret, frame = self.cap.read()
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rotated_frame

    def check_activation(self):
        return self.cap.isOpened()

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()


class TimestampPoster:
    def __init__(self):
        self.post_url = json.load(open(JSON_PATH, 'r'))["post_url"]

    def post_timestamp(self):
        requests.post(self.post_url,
                      json.dumps({'timestamp': str(int(time.time()))}),
                      headers={'Content-Type': 'application/json'})
        print('timestamp has posted:', str(int(time.time())))


class MarkerDetector:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def detect_markers(self, frame, binary_threshold=100):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame_binary = cv2.threshold(frame_gray, binary_threshold, 255, cv2.THRESH_BINARY)
        corners, ids, _ = cv2.aruco.detectMarkers(frame_binary, self.dictionary)
        markers = []
        if ids is not None:
            for i, c in zip(ids, corners):
                marker = Marker(id=i[0], left_top=tuple(c[0].astype(np.int)[0]), right_bottom=tuple(c[0].astype(np.int)[2]))
                markers.append(marker)
                cv2.rectangle(frame, marker.left_top, marker.right_bottom, (0, 255, 0), 1)
        cv2.imshow('markers', frame)
        return markers


class CigarCounter:
    def __init__(self, camera):
        self.camera = camera
        self.marker_detector = MarkerDetector()

    def count(self):
        frame = self.camera.read_frame()
        markers = self.marker_detector.detect_markers(frame)
        left_marker, right_marker = self.get_cigar_markers(markers)
        ret, count = self.count_cigars(frame, left_marker, right_marker)
        print('count', count)
        return ret, count

    @staticmethod
    def get_cigar_markers(markers):
        left_marker = None
        right_marker = None
        for m in markers:
            if m.id == 1:
                right_marker = m
            elif m.id == 2:
                left_marker = m
        return left_marker, right_marker

    @staticmethod
    def count_cigars(frame, left_marker, right_marker, cigar_num=10, binary_threshold=100):
        if left_marker is None or right_marker is None:
            return False, None
        lx1, ly1 = left_marker.left_top
        lx2, ly2 = left_marker.right_bottom
        rx2, ry2 = right_marker.right_bottom
        marker_height = abs(ly2 - ly1)
        y = int(marker_height * 2 / 3)
        count = 0
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame_binary = cv2.threshold(frame_gray, binary_threshold, 255, cv2.THRESH_BINARY)
        frame_binary_roi = frame_binary[ly1-marker_height: ly1, lx1: rx2+1]
        frame_binary_color_roi = cv2.cvtColor(frame_binary_roi, cv2.COLOR_GRAY2BGR)
        for i in range(cigar_num):
            x = int((rx2 - lx1)/127 * (5 + 13 * i))
            cv2.drawMarker(frame_binary_color_roi, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=5)
            if frame_binary_roi[y, x] == 255:
                count += 1
        cv2.putText(frame_binary_color_roi, "count: {:}".format(count), (0, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 255, 0), thickness=1)
        cv2.imshow('roi', frame_binary_color_roi)
        return True, count


class DecrementDetector:
    def __init__(self, counter, log_length=5):
        self.counter = counter
        self.count_log = [-1] * log_length
        self.count = 0

    def judge_decrement(self):
        ret, count = self.counter.count()
        if not ret:
            return False
        self.count_log = self.count_log[1:] + [count]
        if len(set(self.count_log)) == 1:
            if self.count - count == 1:
                self.count = count
                return True
            self.count = count
        return False

    def get_count(self):
        return self.count


class CigarTimingPoster:
    def __init__(self):
        self.camera = Camera()
        counter = CigarCounter(self.camera)
        self.decrement_detector = DecrementDetector(counter)
        self.timestamp_poster = TimestampPoster()
        self.slack_poster = SlackPoster()
        self.slack_post_flag = True
        self.checked_hour = datetime.datetime.now().hour
        self.detect_timing()

    def detect_timing(self, check_hour=10):
        while self.camera.check_activation():
            if self.decrement_detector.judge_decrement():
                self.timestamp_poster.post_timestamp()
            now_hour = datetime.datetime.now().hour
            if now_hour == check_hour and self.checked_hour == check_hour - 1:
                self.slack_post_flag = True
            self.checked_hour = now_hour
            if self.decrement_detector.get_count() == 0 and self.slack_post_flag and now_hour == check_hour:
                self.slack_poster.post_message("<@U9V598VFX> タバコをセットしてください")
                self.slack_post_flag = False
            if cv2.waitKey(1) != -1:
                break
        self.camera.stop()


if __name__ == '__main__':
    CigarTimingPoster()


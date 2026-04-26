# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress TF oneDNN info warning

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TF Python deprecation warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

tf.compat.v1.disable_eager_execution()
# import keras.backend.tensorflow_backend as KTF  # Removed for TF 2.x

from timeit import time
import warnings
import argparse

import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator
import threading
from concurrent.futures import ThreadPoolExecutor

from yolo_v3 import YOLO3
from yolo_v4 import YOLO4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from reid import REID, compute_blur_score, COS_THRESH_HQ, COS_THRESH_LQ, BLUR_THRESHOLD
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
parser.add_argument('--videos', nargs='+', help='List of videos')
parser.add_argument('--webcam', help='Webcam index', type=int)
parser.add_argument('--webcams', nargs='+', help='List of webcams')
parser.add_argument('-all', help='Combine all videos into one', default=True)
args = parser.parse_args()


class WebcamStream:
    def __init__(self, src=0, max_retries=30, retry_delay=0.5):
        print(f'[WebcamStream] Opening source: {src}')
        self.src = src
        self.stream = cv2.VideoCapture(src)
        for attempt in range(max_retries):
            if self.stream.isOpened():
                break
            print(f'[WebcamStream] Retrying connection to {src} ({attempt+1}/{max_retries})...')
            self.stream.release()
            time.sleep(retry_delay)
            self.stream = cv2.VideoCapture(src)
        
        if not self.stream.isOpened():
            raise RuntimeError(f'Cannot open camera/stream: {src}')
        
        # Warmup: local webcams often return black frames on first reads
        warmup = 30 if isinstance(src, int) else 5
        self.grabbed = False
        self.frame = None
        for _ in range(warmup):
            g, f = self.stream.read()
            if g and f is not None:
                self.grabbed, self.frame = g, f
            time.sleep(0.05)

        if not self.grabbed:
            raise RuntimeError(f'Cannot read first frame from: {src}')

        self.w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.frame.shape[1]
        self.h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.frame.shape[0]
        self.fps = int(self.stream.get(cv2.CAP_PROP_FPS)) or 30
        self.stopped = False
        print(f'[WebcamStream] Connected: {src}  res={self.w}x{self.h}  fps={self.fps}')

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            try:
                grabbed, frame = self.stream.read()
                if grabbed and frame is not None:
                    self.grabbed = grabbed
                    self.frame = frame
                else:
                    # Stream dropped a frame — try to reopen
                    time.sleep(0.05)
                    self.stream.release()
                    self.stream = cv2.VideoCapture(self.src)
            except cv2.error:
                # OpenCV C++ exception — sleep and retry without killing the thread
                time.sleep(0.1)
                try:
                    self.stream.release()
                    self.stream = cv2.VideoCapture(self.src)
                except Exception:
                    pass

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path, self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh


def main(yolo):
    print(f'Using {yolo} model')
    # Definition of the parameters
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4

    # deep_sort
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # use to get feature
    
    sess = tf.compat.v1.Session()
    # ReID logic: pre-load images and extract features
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)

    output_frames = []
    output_rectanger = []
    output_areas = []
    output_wh_ratio = []

    is_vis = True
    out_dir = 'videos/output/'
    print('The output folder is', out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    all_frames = []
    for video in args.videos:
        loadvideo = LoadVideo(video)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        while True:
            ret, frame = video_capture.read()
            if ret is not True:
                video_capture.release()
                break
            all_frames.append(frame)

    frame_nums = len(all_frames)
    tracking_path = out_dir + 'tracking' + '.avi'
    combined_path = out_dir + 'allVideos' + '.avi'
    if is_vis:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
        out2 = cv2.VideoWriter(combined_path, fourcc, frame_rate, (w, h))
        # Combine all videos
        for frame in all_frames:
            out2.write(frame)
        out2.release()

    # Initialize tracking file
    filename = out_dir + '/tracking.txt'
    open(filename, 'w')

    fps = 0.0
    frame_cnt = 0
    t1 = time.time()

    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []
    for frame in all_frames:
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)  # n * [topleft_x, topleft_y, w, h]
        features = encoder(frame, boxs)  # n * 128
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]  # length = n
        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
        # indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  # length = len(indices)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        tmp_ids = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                tmp_ids.append(track.track_id)
                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [
                        [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]
                    ]
                    images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt[track.track_id].append([
                        frame_cnt,
                        int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                        area
                    ])
                    images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            cv2_addBox(
                track.track_id,
                frame,
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                line_thickness,
                text_thickness,
                text_scale
            )
            write_results(
                filename,
                'mot',
                frame_cnt + 1,
                str(track.track_id),
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                w, h
            )
        ids_per_frame.append(set(tmp_ids))

        # save a frame
        if is_vis:
            out.write(frame)
        t2 = time.time()

        frame_cnt += 1
        print(frame_cnt, '/', frame_nums)

    if is_vis:
        out.release()
    print('Tracking finished in {} seconds'.format(int(time.time() - t1)))
    print('Tracked video : {}'.format(tracking_path))
    print('Combined video : {}'.format(combined_path))

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    reid = REID()
    final_fuse_id = post_process_reid(reid, images_by_id, ids_per_frame, t1)
    t2 = time.time()

    # To generate MOT for each person, declare 'is_vis' to True
    is_vis = False
    if is_vis:
        print('Writing videos for each ID...')
        output_dir = 'videos/output/tracklets/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        loadvideo = LoadVideo(combined_path)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        for idx in final_fuse_id:
            tracking_path = os.path.join(output_dir, str(idx)+'.avi')
            out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
            for i in final_fuse_id[idx]:
                for f in track_cnt[i]:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, f[0])
                    _, frame = video_capture.read()
                    text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                    cv2_addBox(idx, frame, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                    out.write(frame)
            out.release()
        video_capture.release()

    # Generate a single video with complete MOT/ReID
    if args.all:
        loadvideo = LoadVideo(combined_path)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        complete_path = out_dir+'/Complete'+'.avi'
        out = cv2.VideoWriter(complete_path, fourcc, frame_rate, (w, h))

        for frame in range(len(all_frames)):
            frame2 = all_frames[frame]
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            _, frame2 = video_capture.read()
            for idx in final_fuse_id:
                for i in final_fuse_id[idx]:
                    for f in track_cnt[i]:
                        # print('frame {} f0 {}'.format(frame,f[0]))
                        if frame == f[0]:
                            text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                            cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
            out.write(frame2)
        out.release()
        video_capture.release()

    os.remove(combined_path)
    print('\nWriting videos took {} seconds'.format(int(time.time() - t2)))
    print('Final video at {}'.format(complete_path))
    print('Total: {} seconds'.format(int(time.time() - t1)))


def post_process_reid(reid, images_by_id, ids_per_frame, t1,
                      blur_scores_by_id=None):
    """Match track IDs across cameras using EMA embeddings + quality-aware thresholds.

    When ``blur_scores_by_id`` is provided:
      • Each track's embedding is computed via EMA, skipping blurry frames.
      • Cross-camera cosine distance threshold is loosened (COS_THRESH_LQ) if
        either the new or the existing track has a low mean blur score.
    Falls back to the old behaviour (mean cosine distance, fixed threshold) when
    no quality info is supplied.
    """
    exist_ids     = set()
    final_fuse_id = dict()

    print(f'Total IDs = {len(images_by_id)}')
    feats      = dict()
    mean_blur  = dict()   # track_id -> float  (mean Laplacian variance)

    for tid in images_by_id:
        crops  = images_by_id[tid]
        scores = (blur_scores_by_id or {}).get(tid, [BLUR_THRESHOLD] * len(crops))
        mean_blur[tid] = float(np.mean(scores)) if scores else 0.0
        print(f'ID {tid} -> frames={len(crops)}  mean_blur={mean_blur[tid]:.1f}')

        if blur_scores_by_id:
            feats[tid] = reid._features_with_quality(crops, scores)
        else:
            feats[tid] = reid._features(crops)

    for f in ids_per_frame:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    if len(images_by_id[nid]) < 10:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key, item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))

                    nid_hq = mean_blur.get(nid, BLUR_THRESHOLD) >= BLUR_THRESHOLD
                    for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                        oid_hq = mean_blur.get(oid, BLUR_THRESHOLD) >= BLUR_THRESHOLD
                        # Adaptive cosine threshold: lenient when either side is blurry
                        threshold = COS_THRESH_HQ if (nid_hq and oid_hq) else COS_THRESH_LQ
                        dist = float(np.mean(reid.compute_distance(feats[nid], feats[oid])))
                        print(f'nid={nid} oid={oid} dist={dist:.4f} thresh={threshold:.2f} '
                              f'nid_hq={nid_hq} oid_hq={oid_hq}')
                        dis.append([oid, dist, threshold])

                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue

                    dis.sort(key=operator.itemgetter(1))
                    best_oid, best_dist, best_thresh = dis[0]
                    if best_dist < best_thresh:
                        images_by_id[best_oid] += images_by_id[nid]
                        final_fuse_id[best_oid].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]

    print('Final ids and their sub-ids:', final_fuse_id)
    print('MOT took {} seconds'.format(int(time.time() - t1)))
    return final_fuse_id



def run_webcam_mode(yolo, camera_index=0):
    print(f'Using {yolo} model with webcam index {camera_index}')
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4

    # deep_sort
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)

    out_dir = 'videos/output/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {camera_index}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_val = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    tracking_path = out_dir + 'tracking_webcam' + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(tracking_path, fourcc, fps_val, (w, h))

    filename = out_dir + '/tracking_webcam.txt'
    open(filename, 'w')

    t1 = time.time()
    frame_cnt = 0
    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []

    print("Press 'q' to stop webcam mode")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(frame[..., ::-1])
        boxs = yolo.detect_image(image)
        features = encoder(frame, boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)
        tmp_ids = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                tmp_ids.append(track.track_id)
                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                    images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt[track.track_id].append([frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
                    images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            
            cv2_addBox(track.track_id, frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), line_thickness, text_thickness, text_scale)
            write_results(filename, 'mot', frame_cnt + 1, str(track.track_id), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), w, h)
        
        ids_per_frame.append(set(tmp_ids))
        out.write(frame)
        cv2.imshow('Tracking', frame)
        frame_cnt += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('Tracking finished. Starting ReID post-processing...')
    reid = REID()
    final_fuse_id = post_process_reid(reid, images_by_id, ids_per_frame, t1)
    print('Final ids and their sub-ids:', final_fuse_id)
    print('Webcam tracking saved to {}'.format(tracking_path))
    print('Webcam tracking log saved to {}'.format(filename))


def run_multi_webcam_mode(yolo, cameras):
    print(f'Using {yolo} model with multi webcams: {cameras}')
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4

    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    streams = []
    trackers = []
    for cam in cameras:
        try:
            cam_src = int(cam)
        except ValueError:
            cam_src = cam
            
        stream = WebcamStream(cam_src).start()
        streams.append(stream)
        
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric, max_age=100)
        trackers.append(tracker)

    out_dir = 'videos/output/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    all_w = sum([s.w for s in streams])
    max_h = max([s.h for s in streams]) if streams else 480
    fps_val = streams[0].fps if streams else 30

    tracking_path = os.path.join(out_dir, 'tracking_multi_webcam.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(tracking_path, fourcc, fps_val, (all_w, max_h))

    filename = out_dir + '/tracking_multi_webcam.txt'
    open(filename, 'w')

    t1 = time.time()
    frame_cnt = 0
    track_cnt         = dict()
    images_by_id      = dict()
    blur_scores_by_id = dict()   # track_id -> [float]  Laplacian variance per crop
    ids_per_frame     = []
    
    print("Press 'q' to stop multi webcam mode")
    
    # TF1 sessions are NOT thread-safe and K.learning_phase() resolves
    # to the current thread's default graph — both cause crashes in workers.
    # Lock serializes all sess.run() calls; graph context fixes the tensor lookup.
    tf_lock = threading.Lock()
    yolo_graph = yolo.sess.graph
    yolo_sess  = yolo.sess
    
    def process_cam(i, frame, current_frame_cnt):
        orig_frame = frame.copy()
        image = Image.fromarray(frame[..., ::-1])
        # Run YOLO+encoder under the lock inside the correct TF graph/session
        # context so K.learning_phase() resolves to the right graph on any thread.
        with tf_lock:
            with yolo_graph.as_default():
                with yolo_sess.as_default():
                    boxs = yolo.detect_image(image)
                    features = encoder(frame, boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
        
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
        detections = [detections[idx] for idx in indices]
        
        trackers[i].predict()
        trackers[i].update(detections)
        cam_prefix = f"cam{i}_"
        
        local_tmp_ids      = []
        local_track_cnt    = {}
        local_images_by_id = {}
        local_blur_scores  = {}      # track_id -> float

        for track in trackers[i].tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            track_id = f"{cam_prefix}{track.track_id}"
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < frame.shape[0] and bbox[2] < frame.shape[1]:
                local_tmp_ids.append(track_id)
                tdata = [current_frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]
                local_track_cnt[track_id] = tdata
                crop = orig_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                local_images_by_id[track_id] = crop
                # ── Blur detection ─────────────────────────────────────────
                blur_var = compute_blur_score(crop)
                local_blur_scores[track_id] = blur_var
                is_blurry = blur_var < BLUR_THRESHOLD
                # Draw box: orange label for blurry, normal color otherwise
                box_color = (0, 165, 255) if is_blurry else None
                cv2_addBox(track_id, frame,
                           int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                           line_thickness, text_thickness, text_scale,
                           override_color=box_color)
                # Optional blur-score debug overlay
                cv2.putText(frame, f"{blur_var:.0f}",
                            (int(bbox[0]), int(bbox[1]) - 4),
                            cv2.FONT_HERSHEY_PLAIN, text_scale * 0.85,
                            (255, 255, 0), thickness=1)

        cv2.putText(frame, f"Cam {i}", (10, 30), cv2.FONT_HERSHEY_PLAIN, text_scale + 1, (0, 255, 0), thickness=text_thickness + 1)
        scale = max_h / frame.shape[0] if frame.shape[0] > 0 else 1.0
        new_w = int(frame.shape[1] * scale)
        display_frame = cv2.resize(frame, (new_w, max_h))
        return i, display_frame, local_tmp_ids, local_track_cnt, local_images_by_id, local_blur_scores

    executor = ThreadPoolExecutor(max_workers=len(streams) if streams else 1)

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 60  # ~2 seconds at 30fps before giving up

    while True:
        frames = []      # list of (cam_index, frame)
        all_ok = True
        for i, s in enumerate(streams):
            ret, frame = s.read()
            if ret and frame is not None:
                frames.append((i, frame))
            else:
                all_ok = False
                print(f'[Warning] Frame dropped on Cam {i}, skipping...')

        if not frames:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print('All streams failed for too long. Exiting.')
                break
            time.sleep(0.03)
            continue
        consecutive_failures = 0
            
        futures = []
        for cam_idx, frame in frames:
            futures.append((cam_idx, executor.submit(process_cam, cam_idx, frame, frame_cnt)))
            
        display_frames = [None] * len(streams)
        tmp_ids = []
        
        for cam_idx, future in futures:
            i, display_frame, local_tmp_ids, local_track_cnt, local_images_by_id, local_blur_scores = future.result()
            display_frames[i] = display_frame
            tmp_ids.extend(local_tmp_ids)
            
            src_frame = next(f for ci, f in frames if ci == i)
            for tid, tdata in local_track_cnt.items():
                if tid not in track_cnt:
                    track_cnt[tid]        = [tdata]
                    images_by_id[tid]     = [local_images_by_id[tid]]
                    blur_scores_by_id[tid] = [local_blur_scores.get(tid, 0.0)]
                else:
                    track_cnt[tid].append(tdata)
                    images_by_id[tid].append(local_images_by_id[tid])
                    blur_scores_by_id[tid].append(local_blur_scores.get(tid, 0.0))
                write_results(filename, 'mot', frame_cnt + 1, tid, tdata[1], tdata[2], tdata[3], tdata[4], src_frame.shape[1], src_frame.shape[0])

        ids_per_frame.append(set(tmp_ids))
        
        if display_frames:
            for i, df in enumerate(display_frames):
                if df is not None:
                    cv2.namedWindow(f'Cam {i}', cv2.WINDOW_NORMAL)
                    cv2.imshow(f'Cam {i}', df)
            
            combined_img = cv2.hconcat(display_frames)
            try:
                out.write(cv2.resize(combined_img, (all_w, max_h)))
            except Exception as e:
                pass
            
        frame_cnt += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    executor.shutdown()
    for s in streams:
        s.stop()
    out.release()
    cv2.destroyAllWindows()

    print('Tracking finished. Starting ReID post-processing...')
    reid = REID()
    final_fuse_id = post_process_reid(reid, images_by_id, ids_per_frame, t1,
                                      blur_scores_by_id=blur_scores_by_id)
    print('Final ids and their sub-ids:', final_fuse_id)
    print('Multi Webcam tracking saved to {}'.format(tracking_path))
    print('Multi Webcam tracking log saved to {}'.format(filename))


def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale, override_color=None):
    if isinstance(track_id, str):
        idx = sum([ord(c) for c in str(track_id)])
    else:
        idx = abs(track_id)
    color = override_color if override_color is not None else get_color(idx)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))


warnings.filterwarnings('ignore')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


if __name__ == '__main__':
    # Fix for newer TF 2.x session handling
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    
    yolo_model = YOLO3(sess=sess) if args.version == 'v3' else YOLO4(sess=sess)
    
    if args.webcams is not None:
        run_multi_webcam_mode(yolo=yolo_model, cameras=args.webcams)
    elif args.webcam is not None:
        run_webcam_mode(yolo=yolo_model, camera_index=args.webcam)
    elif args.videos:
        main(yolo=yolo_model)
    else:
        print("Error: Please provide either --videos, --webcam, or --webcams argument.")

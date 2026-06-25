import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from collections import deque
import os
import yaml
import importlib.resources

from .utils import keep_largest_component


class YOLOSegProcessor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        fast_mode: bool = False,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_class_ids: Optional[List[int]] = None,
        device: str = "0",
        enable_tracking: bool = True,
        tracker_config: Optional[str] = None,
        compensate_frames: int = 2,
        mask_smooth: bool = True,
        ignore_class_ids: Optional[List[int]] = [1, 2],
        activation_frames: int = 10,
    ):
        # 处理默认模型路径（从包内 checkpoints 加载）
        if model_path is None:
            # 根据 fast 选择模型文件名
            model_filename = "Yoloseg26n_best.pt" if fast_mode else "Yoloseg26s_best.pt"
            with importlib.resources.path('package_seg.checkpoints', model_filename) as p:
                model_path = str(p)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_class_ids = target_class_ids
        self.device = device
        self.enable_tracking = enable_tracking
        self.mask_smooth = mask_smooth
        self.compensate_frames = compensate_frames
        self.ignore_class_ids = ignore_class_ids or []
        self.activation_frames = activation_frames

        self.colors = [
            (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
            (255,128,0),(128,0,255),(0,128,255),(128,255,0),(255,80,80),(80,255,80),
            (80,80,255),(255,180,0),(180,0,255)
        ]

        # 配置文件路径：优先使用传入，否则从包内 configs 目录查找
        if tracker_config is not None:
            self.tracker_config = tracker_config
        else:
            try:
                with importlib.resources.path('package_seg.configs', 'my_botsort.yaml') as p:
                    if p.exists():
                        self.tracker_config = str(p)
                    else:
                        self.tracker_config = None
            except:
                self.tracker_config = None

        # 跟踪历史
        self.track_history = {}
        self.track_last_mask = {}
        self.track_last_box = {}
        self.track_miss_count = {}
        self.track_start_order = {}
        self.next_order = 0
        self.track_class = {}
        self.track_activation_count = {}
        self.track_is_active = {}
        self._first_frame = True

    def _run_detection_and_tracking(self, frame_rgb: np.ndarray):
        H, W = frame_rgb.shape[:2]
        if self.enable_tracking:
            results = self.model.track(frame_rgb, persist=True, tracker=self.tracker_config,
                                       conf=self.conf_threshold, iou=self.iou_threshold,
                                       verbose=False, device=self.device)
        else:
            results = self.model(frame_rgb, conf=self.conf_threshold, iou=self.iou_threshold,
                                 verbose=False, device=self.device)

        detections = []
        if results and results[0].masks is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy()
            if self.enable_tracking and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.full(len(boxes), -1, dtype=int)

            for i, cls in enumerate(class_ids):
                if self.target_class_ids is not None and cls not in self.target_class_ids:
                    continue
                if scores[i] < self.conf_threshold:
                    continue
                mask = masks[i]
                mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                detections.append({
                    "mask": binary_mask,
                    "box": boxes[i].tolist(),
                    "score": float(scores[i]),
                    "class_id": int(cls),
                    "track_id": int(track_ids[i]),
                })
        return detections

    def _compute_velocity(self, history):
        if len(history) < 2:
            return (0.0, 0.0)
        (cx1, cy1, _, _) = history[-2]
        (cx2, cy2, _, _) = history[-1]
        return (cx2 - cx1, cy2 - cy1)

    def _apply_mask_translation(self, mask, vx, vy):
        if vx == 0 and vy == 0:
            return mask
        H, W = mask.shape[:2]
        M = np.float32([[1, 0, vx], [0, 1, vy]])
        shifted = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return shifted

    def _draw_overlay(self, frame_rgb: np.ndarray, detections: List[Dict],
                      alpha: float = 0.45, draw_id_text: bool = True) -> np.ndarray:
        active_dets = [d for d in detections if self.track_is_active.get(d.get('track_id', -1), False)]
        if self.ignore_class_ids:
            filtered_dets = [d for d in active_dets if d.get('class_id', -1) not in self.ignore_class_ids]
        else:
            filtered_dets = active_dets

        sorted_dets = sorted(filtered_dets, key=lambda d: self.track_start_order.get(d.get('track_id', -1), 0))
        color_layer = frame_rgb.copy()
        masks_info = []
        for det in sorted_dets:
            tid = det.get('track_id', -1)
            if tid == -1:
                continue
            mask = det['mask']
            if mask is None or mask.sum() == 0:
                continue
            if self.mask_smooth:
                mask = keep_largest_component(mask)
            color = self.colors[tid % len(self.colors)]
            color_layer[mask > 0] = color
            masks_info.append((mask, color, tid))

        result = frame_rgb.copy()
        cv2.addWeighted(frame_rgb, 1 - alpha, color_layer, alpha, 0, dst=result)

        if draw_id_text:
            for mask, color, tid in masks_info:
                if np.any(mask > 0):
                    ys, xs = np.where(mask > 0)
                    x_text, y_text = int(xs.min()), int(max(ys.min() - 5, 0))
                    cv2.putText(result, f"ID{tid}", (x_text, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    cv2.putText(result, f"ID{tid}", (x_text, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return result

    def process_frame(self, frame: np.ndarray, draw_overlay: bool = True,
                      alpha: float = 0.45, draw_id_text: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        detections = self._run_detection_and_tracking(frame)
        current_tids = set()

        if self._first_frame:
            for det in detections:
                tid = det.get('track_id', -1)
                if tid == -1:
                    continue
                current_tids.add(tid)
                self.track_is_active[tid] = True
                if tid not in self.track_start_order:
                    self.track_start_order[tid] = self.next_order
                    self.next_order += 1
                self.track_activation_count[tid] = self.activation_frames
                self.track_class[tid] = det['class_id']
                cx = (det['box'][0] + det['box'][2]) / 2
                cy = (det['box'][1] + det['box'][3]) / 2
                if tid not in self.track_history:
                    self.track_history[tid] = deque(maxlen=5)
                self.track_history[tid].append((cx, cy, det['mask'].copy(), det['box'].copy()))
                self.track_last_mask[tid] = det['mask'].copy()
                self.track_last_box[tid] = det['box'].copy()
                self.track_miss_count[tid] = 0
            self._first_frame = False
        else:
            for det in detections:
                tid = det.get('track_id', -1)
                if tid == -1:
                    continue
                current_tids.add(tid)

                if not self.track_is_active.get(tid, False):
                    self.track_activation_count[tid] = self.track_activation_count.get(tid, 0) + 1
                    if self.track_activation_count[tid] >= self.activation_frames:
                        self.track_is_active[tid] = True
                        if tid not in self.track_start_order:
                            self.track_start_order[tid] = self.next_order
                            self.next_order += 1
                else:
                    self.track_activation_count[tid] = self.track_activation_count.get(tid, 0) + 1

                self.track_class[tid] = det['class_id']
                cx = (det['box'][0] + det['box'][2]) / 2
                cy = (det['box'][1] + det['box'][3]) / 2
                if tid not in self.track_history:
                    self.track_history[tid] = deque(maxlen=5)
                self.track_history[tid].append((cx, cy, det['mask'].copy(), det['box'].copy()))
                self.track_last_mask[tid] = det['mask'].copy()
                self.track_last_box[tid] = det['box'].copy()
                self.track_miss_count[tid] = 0

        for tid in list(self.track_miss_count.keys()):
            if tid not in current_tids:
                self.track_miss_count[tid] = self.track_miss_count.get(tid, 0) + 1
                if not self.track_is_active.get(tid, False):
                    self.track_activation_count[tid] = 0

        draw_dets = []
        for det in detections:
            draw_dets.append(det)

        # 补偿：使用最近5帧掩码交集
        for tid, miss_cnt in self.track_miss_count.items():
            if tid in current_tids:
                continue
            if miss_cnt > self.compensate_frames:
                continue
            if not self.track_is_active.get(tid, False):
                continue

            hist = self.track_history.get(tid)
            if hist is None or len(hist) == 0:
                continue

            masks = [item[2] for item in list(hist)]
            if len(masks) >= 5:
                pred_mask = masks[-1].copy()
                for m in masks[-2:-6:-1]:
                    pred_mask = cv2.bitwise_and(pred_mask, m)
            else:
                pred_mask = masks[-1].copy()

            last_box = self.track_last_box.get(tid)
            if last_box is None:
                continue
            pred_box = last_box

            class_id = self.track_class.get(tid, -1)
            draw_dets.append({
                "mask": pred_mask,
                "box": pred_box,
                "track_id": tid,
                "score": 0.0,
                "class_id": class_id,
            })

        to_delete = [tid for tid, cnt in self.track_miss_count.items() if cnt > self.compensate_frames]
        for tid in to_delete:
            self.track_history.pop(tid, None)
            self.track_last_mask.pop(tid, None)
            self.track_last_box.pop(tid, None)
            self.track_miss_count.pop(tid, None)
            self.track_start_order.pop(tid, None)
            self.track_class.pop(tid, None)
            self.track_activation_count.pop(tid, None)
            self.track_is_active.pop(tid, None)

        if draw_overlay:
            overlay = self._draw_overlay(frame, draw_dets, alpha, draw_id_text)
        else:
            overlay = frame.copy()
        return overlay, draw_dets

    def process_video(self, video_path: str, show_progress: bool = False,
                      alpha: float = 0.45, draw_id_text: bool = False,
                      output_video_path: Optional[str] = None) -> List[Tuple[np.ndarray, List[Dict]]]:
        """
        处理视频，并可选保存输出视频。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {total_frames} 帧, {width}x{height}, {fps} fps")
        print(f"跟踪: {'启用 (BoT-SORT)' if self.enable_tracking else '禁用'}")
        print(f"补偿帧数: {self.compensate_frames}")
        print(f"机械臂类别(不绘制): {self.ignore_class_ids}")
        print(f"第一帧全部激活，后续新实例需连续 {self.activation_frames} 帧才激活")

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        results = []
        pbar = tqdm(total=total_frames, desc="处理帧") if show_progress else None
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            overlay, detections = self.process_frame(frame_rgb, draw_overlay=True,
                                                     alpha=alpha, draw_id_text=draw_id_text)
            results.append((overlay, detections))
            if writer:
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if pbar:
                pbar.update(1)

        cap.release()
        if writer:
            writer.release()
        if pbar:
            pbar.close()
        return results

    def reset_tracker(self):
        self.track_history.clear()
        self.track_last_mask.clear()
        self.track_last_box.clear()
        self.track_miss_count.clear()
        self.track_start_order.clear()
        self.track_class.clear()
        self.track_activation_count.clear()
        self.track_is_active.clear()
        self.next_order = 0
        self._first_frame = True

# 包裹分割与跟踪工具

本工具集成了针对物流包裹实例分割与跟踪的 YOLO26s‑seg 分割模型和 BoT‑SORT 跟踪器，提供 API 进行单帧/视频流处理，支持掩码补偿、延迟激活、类别过滤等功能。训练数据经由 GroundingDINO + SAM2 + Depth‑Pro 构成的自动标注 Pipeline 生成。本仓库为推理工具包，不包含数据自动标注代码。

## 环境配置

pip install -r requirements.txt

pip install -e .

## 分割工具接口说明

### YOLOSegProcessor 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | Optional[str] | `None` | 自定义模型权重路径；若为 None，则根据 fast_mode 自动加载内置权重 |
| `conf_threshold` | float | `0.25` | 检测置信度阈值，低于此值的预测框将被过滤 |
| `iou_threshold` | float | `0.45` | NMS 时的 IoU 阈值，用于去除重叠框 |
| `target_class_ids` | Optional[List[int]] | `None` | 指定需要分割的类别 ID（ `[0]` 只分割包裹，`None` 表示包含包裹和机械臂等的所有类别） |
| `device` | str | `"0"` | 推理设备，`"0"` 表示第一块 GPU，`"cpu"` 表示 CPU |
| `enable_tracking` | bool | `True` | 是否启用多目标跟踪（BoT‑SORT）|
| `tracker_config` | Optional[str] | `None` | 自定义跟踪器配置文件路径；默认使用包内 configs/my_botsort.yaml |
| `compensate_frames` | int | `2` | 目标丢失后，最多补偿的帧数（利用最近 5 帧掩码交集预测位置） |
| `mask_smooth` | bool | `True` | 是否只保留掩码的最大连通域（去除小噪点） |
| `ignore_class_ids` | Optional[List[int]] | `[1, 2]` | 不绘制的类别 ID（`[0]` 为包裹，`[1]` 为机械臂，`[2]` 为包裹标签） |
| `activation_frames` | int | `10` | 新目标需连续出现多少帧后才会被绘制（抑制闪烁）|
| `fast_mode` | bool | `False` | `True` 使用轻量模型 `Yoloseg26n_best.pt`（速度优先）；`False` 使用标准模型 `Yoloseg26s_best.pt`（精度优先） |


## 使用分割工具

### 单帧处理（保持追踪）

```python
processor = YOLOSegProcessor()
overlay, detections = processor.process_frame(frame_rgb)
```

返回:

    overlay: 叠加后的图像 (RGB)
    detections: 检测结果列表，每个元素包含:
        - 'mask': 二值 mask (uint8, 0/255)
        - 'box': [x1,y1,x2,y2]
        - 'score': float
        - 'class_id': int
        - 'track_id': int

### 帧序列/视频处理

```python
processor = YOLOSegProcessor()
results = processor.process_video(video_path, show_progress=True)
```

返回:

    results: 列表，每个元素对应视频的一帧，每个元素为 (overlay, detections)
        - overlay: 该帧叠加后的图像 (RGB)
        - detections: 该帧的检测结果列表，每个元素格式与单帧处理中的 detections 相同

# cone_detector

Traffic cone detection with a fine-tuned Ultralytics YOLO (default `yolo26n`, 2.4 M params,
5.4 GFLOPs). Subscribes to `sensor_msgs/Image`, publishes `vision_msgs/Detection2DArray`
(class id + score per box) and an optional debug image.

No usable pretrained cone YOLO was found (COCO has no cone class), so the model is fine-tuned from
COCO weights on the two public datasets in `data/`.

## Layout

```
data/                 raw datasets (Roboflow "Traffic Cones", TraCon) + generated data/merged/
scripts/              uv training project: prepare_data.py, train.py, export.py
runs/                 Ultralytics training runs (metrics, plots, weights)
models/               exported models installed with the package (cone_yolo.onnx / .engine)
cone_detector/        ROS 2 node + ONNX Runtime / TensorRT runners
```

## Training

```bash
cd scripts
uv sync                                    # torch 2.5.1+cu118 (works on Pascal GTX 1050)
uv run python prepare_data.py              # -> data/merged/ (851 train / 224 val / 108 test imgs)
uv run python train.py                     # train, eval on test split, export models/cone_yolo.onnx
uv run python train.py --model yolo11n.pt  # other model; any Ultralytics detect checkpoint works
uv run python export.py ../runs/yolo26n/weights/best.pt   # re-export only
```

`prepare_data.py` merges the datasets, converts polygon labels to boxes, keeps TraCon's
cone-free road images as negatives and removes byte-identical duplicates across splits.
To add more data (e.g. frames from robot bags, which will help most), add an entry to `SOURCES`.

Results: `runs/yolo26n/` (val), `runs/yolo26n_test/` (held-out test split).

### Visualizing predictions

```bash
uv run python visualize.py --show                       # test split, GT green / predictions magenta
uv run python visualize.py --source /path/to/frames --show --conf 0.3   # any image folder
```

Steps through images in a window (←/→, q to quit) and writes annotated images and contact sheets
to `runs/visualize/`. Runs on CPU by default, so it works while training uses the GPU.

## Running

```bash
ros2 launch cone_detector cone_detector.launch.py image:=/camera/image
ros2 launch cone_detector cone_detector.launch.py image:=/camera/image decompress:=false \
    model_path:=cone_yolo.engine
```

| Topic | Type | |
|---|---|---|
| `image`, `camera_info` (in) | `Image`, `CameraInfo` | `camera_info` launch arg, default `/odin1/camera_info` |
| `/cone_detector/detections` | `vision_msgs/Detection2DArray` | boxes + score; `results[].pose` = 3D position when located |
| `/cone_detector/cone_poses` | `geometry_msgs/PoseStamped` | one message per located cone |
| `/cone_detector/markers` | `visualization_msgs/MarkerArray` | orange cylinder + "range bearing" label per cone |
| `/cone_detector/debug_image` | `Image` | boxes with score and range |

**Cone position.** Assumes upright cones of height `cone_height` (config, default 0.5 m) and a
roughly level camera. The box's top and bottom centre pixels are undistorted with the camera_info
intrinsics (`plumb_bob`, `rational_polynomial` or `equidistant`; set `rectified: true` for
`image_rect` topics to use `P`), which gives depth `Z = cone_height / (y_bottom − y_top)` in
normalized coordinates. Poses are in the image's optical frame (x right, y down, z forward), at the
cone's mid-height, with their z axis pointing up along the cone. Labels show horizontal range and
bearing (positive = left). Boxes cut by the top or bottom image edge get no position. Expect a few
percent underestimate when looking down at cones (their base's front edge makes the box taller).

The input size is read from the model. The node handles both ONNX output layouts: the default raw
head `[1, 4+nc, anchors]` (NMS done in the node) and `end2end=True` exports `[1, 300, 6]`
(NMS-free).

### Backends

- `.onnx` → ONNX Runtime, CUDA if available else CPU. Needs `onnxruntime-gpu` in the ROS Python
  (see the note in `cone_detector/ort_runner.py` for Pascal GPUs).
- `.engine` → TensorRT. **Recommended on the Jetson** (TensorRT ships with JetPack). Build the
  engine on the Jetson itself, since engines are tied to the GPU and TensorRT version:

  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=models/cone_yolo.onnx --saveEngine=models/cone_yolo.engine --fp16
  ```

  then rebuild the package so the engine gets installed.

## cone_finder: visiting cones with crl_commander

`cone_finder` keeps `crl_commander` in `explore` and sends it to every new cone:

```
EXPLORE --confirmed unchecked cone--> GOTO (cone as /operator_goal) --within approach_distance--> STOP
STOP --stop_duration--> cone marked checked --> EXPLORE
GOTO --goto_timeout--> cone marked checked (unreachable) --> EXPLORE
```

```bash
ros2 launch cone_detector cone_detector.launch.py image:=/camera/image camera_info:=/camera/camera_info
ros2 launch cone_detector cone_finder.launch.py      # crl_commander must already be running
```

- Cone poses from `/cone_detector/cone_poses` are transformed into `map_frame` (`FP_ENU0`, same as
  the commander), so TF must connect the camera frame to it.
- A cone is acted on after `min_detections` (3) detections within `association_radius`. That way a
  single false positive does not send the robot away. The position is a weighted mean that favours
  close-range detections, and the goal is resent when it moves by more than `goal_update_distance`.
  Every new goal makes the commander replan.
- A checked cone excludes a `checked_radius` (3 m) circle from further detections.
- Mode switches go through `/crl_commander/switch_mode`, are retried until acknowledged, and are
  only sent on cone_finder's own state changes. If an operator switches the commander by hand,
  cone_finder does not fight it, but it will switch again on its next transition.
- `~/state` (String) and `~/markers` show checked areas (green), candidates (yellow, orange once
  confirmed) and the current target's approach circle (red).
- crl_commander treats a goal as reached inside a ±`goal_reached_dist` box (2.5 m), then holds
  position. That box reaches 3.5 m at its corners, so with `approach_distance` 3 m the robot can stop
  just outside the circle. `goto_timeout` then ends the visit. Raise `approach_distance` to ~3.6 m or
  lower the commander's thresholds if this happens often.

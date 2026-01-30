# UAV ROS Bag 转 Epona 数据集转换工具

本工具将UAV的EuRoC格式ROS bag数据转换为Epona世界模型可用的数据集格式。

## 目录结构

```
uav_to_epona/
├── convert_uav_to_epona.py   # 主转换脚本
├── config.yaml               # 转换配置文件
├── run_convert.sh            # 一键转换脚本
└── README.md                 # 本文件
```

## 环境要求

### ROS 环境
```bash
# 确保已安装ROS并source环境
source /opt/ros/noetic/setup.bash  # 或其他ROS版本
```

### Python 依赖
```bash
pip install rosbag cv_bridge opencv-python scipy pyyaml tqdm numpy
```

如果`cv_bridge`安装有问题，可以尝试：
```bash
pip install rospkg
# 或从ROS仓库安装
sudo apt-get install ros-noetic-cv-bridge
```

## 快速开始

### 方式一：一键转换所有数据集（推荐）

```bash
cd /home/dataset-local/uav_to_epona
source /opt/ros/noetic/setup.bash  # 确保ROS环境已加载
./run_convert.sh
```

### 方式二：批量转换指定数据集

```bash
# 只转换指定的数据集
PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH" \
    python3 batch_convert.py --datasets MH_01_easy V1_01_easy

# 限制每个数据集的图像数量（用于快速测试）
PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH" \
    python3 batch_convert.py --max_images 500
```

### 方式三：转换单个bag文件

```bash
PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH" \
    python3 convert_uav_to_epona.py \
    --bag_path /path/to/your.bag \
    --output_dir /path/to/output \
    --scene_name MH_01_easy \
    --dataset_type auto
```

## 配置文件说明

`config.yaml` 示例：

```yaml
# 输出目录
output_dir: /home/dataset-local/uav_epona_dataset

# 图像设置
convert_grayscale_to_rgb: true  # 灰度图转RGB
image_quality: 95               # JPEG质量

# 帧率设置
target_fps: 10  # 目标帧率，null保持原始帧率

# 数据集列表
datasets:
  - bag_path: /path/to/MH_01_easy.bag
    scene_name: MH_01_easy
    dataset_type: machine_hall  # 或 vicon_room 或 auto
```

## 输出格式

转换后的数据结构与Epona的nuplan格式兼容：

```
uav_epona_dataset/
├── sensor_blobs/
│   ├── MH_01_easy/
│   │   └── CAM_F0/
│   │       ├── 0000000000000001.jpg
│   │       ├── 0000000000000002.jpg
│   │       └── ...
│   └── ...
├── ego_meta/
│   ├── MH_01_easy.json
│   └── ...
└── test_meta.json
```

### JSON元数据格式

**test_meta.json**：
```json
[
    {
        "CAM_F0": ["0000000000000001.jpg", ...],
        "scene": "MH_01_easy_scene_0",
        "data_root": "sensor_blobs/MH_01_easy",
        "pose": "ego_meta/MH_01_easy.json"
    }
]
```

**ego_meta/*.json**：
```json
{
    "CAM_F0": {
        "CAM_F0/0000000000000001.jpg": {
            "x": 1.234, "y": 5.678, "z": 0.5,
            "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
            "vx": 0.5, "vy": 0.1,
            "ax": 0.01, "ay": 0.02,
            "timestamp": 1403636579810000
        }
    }
}
```

## 在Epona中使用

### 1. 使用nuplan加载器（推荐）

转换后的数据与nuplan格式兼容，可直接使用nuplan配置：

```bash
cd /home/dataset-local/Epona

python3 scripts/test/test_nuplan.py \
    --exp_name "test-uav" \
    --start_id 0 --end_id 1 \
    --resume_path "pretrained/epona_nuplan.pkl" \
    --config configs/dit_config_dcae_uav.py
```

### 2. 使用专用UAV加载器

修改配置文件中的 `train_data_list`：

```python
# 在 dit_config_dcae_uav.py 中
train_data_list = ['uav']
val_data_list = ['uav']

datasets_paths = dict(
    uav_root='/home/dataset-local/uav_epona_dataset',
    uav_json_root='/home/dataset-local/uav_epona_dataset',
)
```

## EuRoC数据集类型

### Machine Hall (MH)
- 位置数据：`/leica/position` (PointStamped)
- 需要从IMU估计姿态

### Vicon Room (V1, V2)
- 完整位姿：`/vicon/firefly_sbx/firefly_sbx` (TransformStamped)
- 包含位置和四元数姿态

## 常见问题

### Q: rosbag包找不到？
```bash
pip install rosbag
# 或
pip install bagpy  # 替代方案
```

### Q: cv_bridge安装失败？
```bash
# 方法1: 使用conda
conda install -c conda-forge ros-cv-bridge

# 方法2: 从源码安装
# 参考: https://github.com/ros-perception/vision_opencv
```

### Q: 图像是灰度的，Epona需要RGB？
转换工具默认会将灰度图转换为RGB（3通道复制）。可以在配置中设置：
```yaml
convert_grayscale_to_rgb: true
```

### Q: 帧率不匹配？
EuRoC约20Hz，nuplan约10Hz。可以在配置中设置目标帧率：
```yaml
target_fps: 10  # 下采样到10Hz
```

## 技术细节

### 位姿数据处理
- 位置(x, y, z)：直接从ROS消息提取
- 姿态(qw, qx, qy, qz)：
  - vicon_room: 直接从TransformStamped提取
  - machine_hall: 从IMU积分估计
- 速度(vx, vy)：从位置差分计算
- 加速度(ax, ay)：从速度差分计算

### 时间对齐
- 图像和位姿时间戳独立记录
- 使用线性插值对齐位姿到图像时间戳
- 四元数使用最近邻匹配

## 许可证

本工具遵循MIT许可证。EuRoC数据集请参考其官方许可。

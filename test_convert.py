#!/usr/bin/env python3
"""
简化测试脚本 - 只处理少量图像来快速验证转换流程
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add ROS path
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')

import rosbag
import cv2
from scipy.interpolate import interp1d

# 配置
BAG_PATH = '/home/dataset-local/uav_datasets/machine_hall/MH_01_easy/MH_01_easy.bag'
OUTPUT_DIR = '/home/dataset-local/uav_epona_dataset'
SCENE_NAME = 'MH_01_easy'
MAX_IMAGES = 500  # 处理500张图像进行测试（约25秒的视频@20Hz）


def imgmsg_to_cv2(msg):
    """手动将ROS Image消息转换为OpenCV图像，避免cv_bridge的库冲突"""
    # 获取图像参数
    height = msg.height
    width = msg.width
    encoding = msg.encoding
    
    # 根据编码确定数据类型
    if encoding in ['mono8', '8UC1']:
        dtype = np.uint8
        channels = 1
    elif encoding in ['mono16', '16UC1']:
        dtype = np.uint16
        channels = 1
    elif encoding in ['bgr8', '8UC3']:
        dtype = np.uint8
        channels = 3
    elif encoding in ['rgb8']:
        dtype = np.uint8
        channels = 3
    else:
        # 默认假设 mono8
        dtype = np.uint8
        channels = 1
    
    # 将数据转换为numpy数组
    if channels == 1:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(height, width)
    else:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, channels)
    
    return img


def main():
    print(f"Testing conversion with {MAX_IMAGES} images...")
    
    # Setup directories
    # 注意：NuPlan加载器对test split使用 test_ego_meta 目录
    sensor_dir = Path(OUTPUT_DIR) / 'sensor_blobs' / SCENE_NAME / 'CAM_F0'
    ego_meta_dir = Path(OUTPUT_DIR) / 'test_ego_meta'
    sensor_dir.mkdir(parents=True, exist_ok=True)
    ego_meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract images and poses
    images_info = []
    poses_raw = []
    
    print("Opening bag file...")
    with rosbag.Bag(BAG_PATH, 'r') as bag:
        # Extract images (limited)
        print("Extracting images...")
        count = 0
        for topic, msg, t in bag.read_messages(topics=['/cam0/image_raw']):
            if count >= MAX_IMAGES:
                break
            
            # Convert image (不使用cv_bridge)
            cv_image = imgmsg_to_cv2(msg)
            
            # Grayscale to RGB
            if len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            
            # Save
            timestamp_ns = msg.header.stamp.to_nsec()
            timestamp_us = timestamp_ns // 1000
            filename = f"{timestamp_us:016d}.jpg"
            filepath = sensor_dir / filename
            cv2.imwrite(str(filepath), cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            images_info.append((timestamp_us, filename))
            count += 1
            
            if count % 20 == 0:
                print(f"  Extracted {count} images...")
        
        print(f"Total images extracted: {len(images_info)}")
        
        # Extract poses
        print("Extracting poses...")
        for topic, msg, t in bag.read_messages(topics=['/leica/position']):
            timestamp_ns = msg.header.stamp.to_nsec()
            timestamp_us = timestamp_ns // 1000
            poses_raw.append({
                'timestamp': timestamp_us,
                'x': msg.point.x,
                'y': msg.point.y,
                'z': msg.point.z,
            })
        
        print(f"Total poses extracted: {len(poses_raw)}")
    
    # Compute velocities
    print("Computing velocities...")
    for i in range(1, len(poses_raw) - 1):
        dt = (poses_raw[i+1]['timestamp'] - poses_raw[i-1]['timestamp']) / 1e6
        if dt > 0:
            poses_raw[i]['vx'] = (poses_raw[i+1]['x'] - poses_raw[i-1]['x']) / dt
            poses_raw[i]['vy'] = (poses_raw[i+1]['y'] - poses_raw[i-1]['y']) / dt
        else:
            poses_raw[i]['vx'] = 0
            poses_raw[i]['vy'] = 0
    
    if len(poses_raw) > 0:
        poses_raw[0]['vx'] = poses_raw[1].get('vx', 0) if len(poses_raw) > 1 else 0
        poses_raw[0]['vy'] = poses_raw[1].get('vy', 0) if len(poses_raw) > 1 else 0
        poses_raw[-1]['vx'] = poses_raw[-2].get('vx', 0) if len(poses_raw) > 1 else 0
        poses_raw[-1]['vy'] = poses_raw[-2].get('vy', 0) if len(poses_raw) > 1 else 0
    
    # Interpolate poses to image timestamps
    print("Interpolating poses...")
    pose_times = np.array([p['timestamp'] for p in poses_raw])
    
    interp_x = interp1d(pose_times, [p['x'] for p in poses_raw], fill_value='extrapolate')
    interp_y = interp1d(pose_times, [p['y'] for p in poses_raw], fill_value='extrapolate')
    interp_z = interp1d(pose_times, [p['z'] for p in poses_raw], fill_value='extrapolate')
    interp_vx = interp1d(pose_times, [p.get('vx', 0) for p in poses_raw], fill_value='extrapolate')
    interp_vy = interp1d(pose_times, [p.get('vy', 0) for p in poses_raw], fill_value='extrapolate')
    
    # Generate ego meta
    ego_meta = {'CAM_F0': {}}
    for timestamp_us, filename in images_info:
        key = f'CAM_F0/{filename}'
        ego_meta['CAM_F0'][key] = {
            'x': float(interp_x(timestamp_us)),
            'y': float(interp_y(timestamp_us)),
            'z': float(interp_z(timestamp_us)),
            'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0,  # Identity for machine_hall
            'vx': float(interp_vx(timestamp_us)),
            'vy': float(interp_vy(timestamp_us)),
            'ax': 0.0, 'ay': 0.0,
            'timestamp': int(timestamp_us),
        }
    
    # Save ego meta
    ego_meta_path = ego_meta_dir / f'{SCENE_NAME}.json'
    with open(ego_meta_path, 'w') as f:
        json.dump(ego_meta, f)
    print(f"Saved ego meta to: {ego_meta_path}")
    
    # Generate test_meta.json
    test_meta = [{
        'CAM_F0': [img[1] for img in images_info],
        'scene': f'{SCENE_NAME}_scene_0',
        'data_root': f'sensor_blobs/{SCENE_NAME}',
        'pose': f'ego_meta/{SCENE_NAME}.json',
    }]
    
    meta_path = Path(OUTPUT_DIR) / 'test_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(test_meta, f, indent=2)
    print(f"Saved test meta to: {meta_path}")
    
    # Verify
    print("\n=== Verification ===")
    print(f"Images saved: {len(list(sensor_dir.glob('*.jpg')))}")
    print(f"Ego meta entries: {len(ego_meta['CAM_F0'])}")
    print(f"Test meta sequences: {len(test_meta)}")
    
    # Show sample
    print("\nSample ego meta entry:")
    sample_key = list(ego_meta['CAM_F0'].keys())[0]
    print(f"  {sample_key}: {ego_meta['CAM_F0'][sample_key]}")
    
    print("\n=== Test completed successfully! ===")

if __name__ == '__main__':
    main()

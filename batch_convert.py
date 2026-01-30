#!/usr/bin/env python3
"""
UAV ROS Bag 批量转换脚本
将所有UAV数据集转换为Epona格式

用法:
    python3 batch_convert.py                    # 处理所有数据集
    python3 batch_convert.py --max_images 500   # 每个数据集只处理500张图像
    python3 batch_convert.py --datasets MH_01_easy MH_02_easy  # 只处理指定数据集
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add ROS path
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')

import rosbag
import cv2
from scipy.interpolate import interp1d
from tqdm import tqdm


# 数据集配置
DATASETS_CONFIG = {
    # Machine Hall
    'MH_01_easy': {
        'bag_path': '/home/dataset-local/uav_datasets/machine_hall/MH_01_easy/MH_01_easy.bag',
        'type': 'machine_hall',
    },
    'MH_02_easy': {
        'bag_path': '/home/dataset-local/uav_datasets/machine_hall/MH_02_easy/MH_02_easy.bag',
        'type': 'machine_hall',
    },
    'MH_03_medium': {
        'bag_path': '/home/dataset-local/uav_datasets/machine_hall/MH_03_medium/MH_03_medium.bag',
        'type': 'machine_hall',
    },
    'MH_04_difficult': {
        'bag_path': '/home/dataset-local/uav_datasets/machine_hall/MH_04_difficult/MH_04_difficult.bag',
        'type': 'machine_hall',
    },
    'MH_05_difficult': {
        'bag_path': '/home/dataset-local/uav_datasets/machine_hall/MH_05_difficult/MH_05_difficult.bag',
        'type': 'machine_hall',
    },
    # Vicon Room 1
    'V1_01_easy': {
        'bag_path': '/home/dataset-local/uav_datasets/vicon_room1/V1_01_easy/V1_01_easy.bag',
        'type': 'vicon_room',
    },
    'V1_02_medium': {
        'bag_path': '/home/dataset-local/uav_datasets/vicon_room1/V1_02_medium/V1_02_medium.bag',
        'type': 'vicon_room',
    },
    'V1_03_difficult': {
        'bag_path': '/home/dataset-local/uav_datasets/vicon_room1/V1_03_difficult/V1_03_difficult.bag',
        'type': 'vicon_room',
    },
}

OUTPUT_DIR = '/home/dataset-local/uav_epona_dataset'


def imgmsg_to_cv2(msg):
    """将ROS Image消息转换为OpenCV图像"""
    height = msg.height
    width = msg.width
    encoding = msg.encoding
    
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
        dtype = np.uint8
        channels = 1
    
    if channels == 1:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(height, width)
    else:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, channels)
    
    return img


def convert_single_dataset(
    scene_name: str,
    bag_path: str,
    dataset_type: str,
    output_dir: str,
    max_images: Optional[int] = None,
) -> Dict:
    """转换单个数据集"""
    print(f"\n{'='*60}")
    print(f"Converting: {scene_name}")
    print(f"Bag: {bag_path}")
    print(f"Type: {dataset_type}")
    print(f"{'='*60}")
    
    if not os.path.exists(bag_path):
        print(f"  [SKIP] Bag file not found: {bag_path}")
        return None
    
    # Setup directories
    # 注意：NuPlan加载器对test split使用 test_ego_meta 目录
    sensor_dir = Path(output_dir) / 'sensor_blobs' / scene_name / 'CAM_F0'
    ego_meta_dir = Path(output_dir) / 'test_ego_meta'
    sensor_dir.mkdir(parents=True, exist_ok=True)
    ego_meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Topic names
    if dataset_type == 'machine_hall':
        pose_topic = '/leica/position'
    else:  # vicon_room
        pose_topic = '/vicon/firefly_sbx/firefly_sbx'
    
    images_info = []
    poses_raw = []
    
    print("  Opening bag file...")
    with rosbag.Bag(bag_path, 'r') as bag:
        # Get message counts
        img_count = bag.get_message_count('/cam0/image_raw')
        pose_count = bag.get_message_count(pose_topic)
        
        if max_images:
            img_count = min(img_count, max_images)
        
        print(f"  Images to extract: {img_count}")
        print(f"  Poses available: {pose_count}")
        
        # Extract images
        print("  Extracting images...")
        count = 0
        for topic, msg, t in tqdm(bag.read_messages(topics=['/cam0/image_raw']), 
                                   total=img_count, desc="  Images"):
            if max_images and count >= max_images:
                break
            
            cv_image = imgmsg_to_cv2(msg)
            
            # Grayscale to RGB
            if len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            
            timestamp_ns = msg.header.stamp.to_nsec()
            timestamp_us = timestamp_ns // 1000
            filename = f"{timestamp_us}.jpg"
            
            filepath = sensor_dir / filename
            cv2.imwrite(str(filepath), cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            images_info.append((timestamp_us, filename))
            count += 1
        
        # Extract poses
        print("  Extracting poses...")
        for topic, msg, t in tqdm(bag.read_messages(topics=[pose_topic]),
                                   total=pose_count, desc="  Poses"):
            timestamp_ns = msg.header.stamp.to_nsec()
            timestamp_us = timestamp_ns // 1000
            
            if dataset_type == 'machine_hall':
                # PointStamped
                poses_raw.append({
                    'timestamp': timestamp_us,
                    'x': msg.point.x,
                    'y': msg.point.y,
                    'z': msg.point.z,
                    'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0,
                })
            else:
                # TransformStamped
                poses_raw.append({
                    'timestamp': timestamp_us,
                    'x': msg.transform.translation.x,
                    'y': msg.transform.translation.y,
                    'z': msg.transform.translation.z,
                    'qw': msg.transform.rotation.w,
                    'qx': msg.transform.rotation.x,
                    'qy': msg.transform.rotation.y,
                    'qz': msg.transform.rotation.z,
                })
    
    if len(images_info) == 0 or len(poses_raw) == 0:
        print("  [ERROR] No data extracted!")
        return None
    
    # Compute velocities
    print("  Computing velocities...")
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
    
    # Interpolate poses
    print("  Interpolating poses to image timestamps...")
    pose_times = np.array([p['timestamp'] for p in poses_raw])
    
    interp_funcs = {
        'x': interp1d(pose_times, [p['x'] for p in poses_raw], fill_value='extrapolate'),
        'y': interp1d(pose_times, [p['y'] for p in poses_raw], fill_value='extrapolate'),
        'z': interp1d(pose_times, [p['z'] for p in poses_raw], fill_value='extrapolate'),
        'qw': interp1d(pose_times, [p['qw'] for p in poses_raw], fill_value='extrapolate'),
        'qx': interp1d(pose_times, [p['qx'] for p in poses_raw], fill_value='extrapolate'),
        'qy': interp1d(pose_times, [p['qy'] for p in poses_raw], fill_value='extrapolate'),
        'qz': interp1d(pose_times, [p['qz'] for p in poses_raw], fill_value='extrapolate'),
        'vx': interp1d(pose_times, [p.get('vx', 0) for p in poses_raw], fill_value='extrapolate'),
        'vy': interp1d(pose_times, [p.get('vy', 0) for p in poses_raw], fill_value='extrapolate'),
    }
    
    # Generate ego meta
    ego_meta = {'CAM_F0': {}}
    for timestamp_us, filename in images_info:
        key = f'CAM_F0/{filename}'
        ego_meta['CAM_F0'][key] = {
            'x': float(interp_funcs['x'](timestamp_us)),
            'y': float(interp_funcs['y'](timestamp_us)),
            'z': float(interp_funcs['z'](timestamp_us)),
            'qw': float(interp_funcs['qw'](timestamp_us)),
            'qx': float(interp_funcs['qx'](timestamp_us)),
            'qy': float(interp_funcs['qy'](timestamp_us)),
            'qz': float(interp_funcs['qz'](timestamp_us)),
            'vx': float(interp_funcs['vx'](timestamp_us)),
            'vy': float(interp_funcs['vy'](timestamp_us)),
            'ax': 0.0,
            'ay': 0.0,
            'timestamp': int(timestamp_us),
        }
    
    # Save ego meta
    ego_meta_path = ego_meta_dir / f'{scene_name}.json'
    with open(ego_meta_path, 'w') as f:
        json.dump(ego_meta, f)
    print(f"  Saved ego meta: {ego_meta_path}")
    
    # Generate sequence meta
    sequence_meta = {
        'CAM_F0': [img[1] for img in images_info],
        'scene': f'{scene_name}_scene_0',
        'data_root': f'sensor_blobs/{scene_name}',
        'pose': f'ego_meta/{scene_name}.json',
    }
    
    print(f"  [OK] Converted {len(images_info)} images, {len(ego_meta['CAM_F0'])} poses")
    
    return sequence_meta


def main():
    parser = argparse.ArgumentParser(description='Batch convert UAV datasets to Epona format')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max images per dataset (None = all)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to convert (e.g., MH_01_easy V1_01_easy)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    args = parser.parse_args()
    
    print("="*60)
    print("UAV to Epona Batch Converter")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Determine which datasets to process
    if args.datasets:
        datasets_to_process = {k: v for k, v in DATASETS_CONFIG.items() if k in args.datasets}
    else:
        datasets_to_process = DATASETS_CONFIG
    
    print(f"\nDatasets to process: {list(datasets_to_process.keys())}")
    if args.max_images:
        print(f"Max images per dataset: {args.max_images}")
    
    # Clear output directory
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        print(f"\nClearing existing output directory: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
    
    # Convert each dataset
    all_sequences = []
    successful = 0
    failed = 0
    
    for scene_name, config in datasets_to_process.items():
        result = convert_single_dataset(
            scene_name=scene_name,
            bag_path=config['bag_path'],
            dataset_type=config['type'],
            output_dir=output_dir,
            max_images=args.max_images,
        )
        
        if result:
            all_sequences.append(result)
            successful += 1
        else:
            failed += 1
    
    # Save combined test_meta.json
    meta_path = Path(output_dir) / 'test_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(all_sequences, f, indent=2)
    print(f"\nSaved combined meta: {meta_path}")
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Output directory: {output_dir}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Verification
    print("\nVERIFICATION:")
    for seq in all_sequences:
        scene = seq['scene'].replace('_scene_0', '')
        img_dir = Path(output_dir) / seq['data_root'] / 'CAM_F0'
        img_count = len(list(img_dir.glob('*.jpg')))
        print(f"  {scene}: {img_count} images, {len(seq['CAM_F0'])} in meta")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
UAV ROS Bag to Epona Dataset Converter

This script converts EuRoC MAV datasets (ROS bag format) to Epona-compatible format.
Supports both machine_hall and vicon_room datasets.

Usage:
    python convert_uav_to_epona.py --config config.yaml
    python convert_uav_to_epona.py --bag_path /path/to/bag --output_dir /path/to/output --scene_name MH_01_easy
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import cv2
from tqdm import tqdm

try:
    import rosbag
except ImportError:
    print("Error: rosbag package not found. Please source your ROS environment.")
    print("Run: source /opt/ros/<distro>/setup.bash")
    sys.exit(1)


def imgmsg_to_cv2(msg):
    """
    手动将ROS Image消息转换为OpenCV图像
    避免cv_bridge的库版本冲突问题
    """
    height = msg.height
    width = msg.width
    encoding = msg.encoding
    
    # 根据编码确定数据类型和通道数
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
    elif encoding in ['bgra8', '8UC4']:
        dtype = np.uint8
        channels = 4
    elif encoding in ['rgba8']:
        dtype = np.uint8
        channels = 4
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


@dataclass
class PoseData:
    """Container for pose data at a single timestamp."""
    timestamp: int  # microseconds
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0


class EurocBagConverter:
    """Converter for EuRoC MAV dataset ROS bags to Epona format."""
    
    # ROS topics for different EuRoC dataset types
    TOPICS = {
        'machine_hall': {
            'cam0': '/cam0/image_raw',
            'cam1': '/cam1/image_raw',
            'imu': '/imu0',
            'pose': '/leica/position',  # geometry_msgs/PointStamped
        },
        'vicon_room': {
            'cam0': '/cam0/image_raw',
            'cam1': '/cam1/image_raw',
            'imu': '/imu0',
            'pose': '/vicon/firefly_sbx/firefly_sbx',  # geometry_msgs/TransformStamped
        }
    }
    
    def __init__(
        self,
        bag_path: str,
        output_dir: str,
        scene_name: str,
        dataset_type: str = 'auto',
        convert_grayscale_to_rgb: bool = True,
        image_quality: int = 95,
        target_fps: Optional[float] = None,
    ):
        """
        Initialize the converter.
        
        Args:
            bag_path: Path to the ROS bag file
            output_dir: Output directory for converted data
            scene_name: Name of the scene (e.g., 'MH_01_easy')
            dataset_type: 'machine_hall', 'vicon_room', or 'auto'
            convert_grayscale_to_rgb: Whether to convert grayscale images to RGB
            image_quality: JPEG quality (1-100)
            target_fps: Target frame rate for downsampling (None = keep original)
        """
        self.bag_path = bag_path
        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        self.convert_grayscale_to_rgb = convert_grayscale_to_rgb
        self.image_quality = image_quality
        self.target_fps = target_fps
        
        # Auto-detect dataset type if needed
        if dataset_type == 'auto':
            self.dataset_type = self._detect_dataset_type()
        else:
            self.dataset_type = dataset_type
        
        print(f"Dataset type: {self.dataset_type}")
        
        # Setup output directories
        self.sensor_blobs_dir = self.output_dir / 'sensor_blobs' / scene_name / 'CAM_F0'
        self.ego_meta_dir = self.output_dir / 'ego_meta'
        
        self.sensor_blobs_dir.mkdir(parents=True, exist_ok=True)
        self.ego_meta_dir.mkdir(parents=True, exist_ok=True)
    
    def _detect_dataset_type(self) -> str:
        """Auto-detect the dataset type from the bag file."""
        with rosbag.Bag(self.bag_path, 'r') as bag:
            topics = bag.get_type_and_topic_info().topics.keys()
            
            if '/leica/position' in topics:
                return 'machine_hall'
            elif '/vicon/firefly_sbx/firefly_sbx' in topics:
                return 'vicon_room'
            else:
                raise ValueError(f"Cannot auto-detect dataset type. Available topics: {list(topics)}")
    
    def _extract_images(self) -> List[Tuple[int, str]]:
        """
        Extract images from the bag file.
        
        Returns:
            List of (timestamp_us, image_filename) tuples
        """
        print("Extracting images...")
        images_info = []
        topics = self.TOPICS[self.dataset_type]
        
        with rosbag.Bag(self.bag_path, 'r') as bag:
            msg_count = bag.get_message_count(topics['cam0'])
            
            for topic, msg, t in tqdm(bag.read_messages(topics=[topics['cam0']]), 
                                       total=msg_count, desc="Extracting images"):
                # Convert ROS image to OpenCV (不使用cv_bridge避免库冲突)
                try:
                    cv_image = imgmsg_to_cv2(msg)
                except Exception as e:
                    print(f"Warning: Failed to convert image at {t}: {e}")
                    continue
                
                # Convert grayscale to RGB if needed
                if self.convert_grayscale_to_rgb and len(cv_image.shape) == 2:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                
                # Generate filename based on timestamp
                timestamp_ns = msg.header.stamp.to_nsec()
                timestamp_us = timestamp_ns // 1000
                filename = f"{timestamp_us:016d}.jpg"
                
                # Save image
                filepath = self.sensor_blobs_dir / filename
                cv2.imwrite(str(filepath), cv_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
                
                images_info.append((timestamp_us, filename))
        
        print(f"Extracted {len(images_info)} images")
        return sorted(images_info, key=lambda x: x[0])
    
    def _extract_poses(self) -> List[PoseData]:
        """
        Extract pose data from the bag file.
        
        Returns:
            List of PoseData objects
        """
        print("Extracting poses...")
        poses = []
        topics = self.TOPICS[self.dataset_type]
        
        with rosbag.Bag(self.bag_path, 'r') as bag:
            msg_count = bag.get_message_count(topics['pose'])
            
            for topic, msg, t in tqdm(bag.read_messages(topics=[topics['pose']]),
                                       total=msg_count, desc="Extracting poses"):
                timestamp_ns = msg.header.stamp.to_nsec()
                timestamp_us = timestamp_ns // 1000
                
                if self.dataset_type == 'machine_hall':
                    # PointStamped: only position available
                    pose = PoseData(
                        timestamp=timestamp_us,
                        x=msg.point.x,
                        y=msg.point.y,
                        z=msg.point.z,
                        qw=1.0, qx=0.0, qy=0.0, qz=0.0  # Identity quaternion
                    )
                else:  # vicon_room
                    # TransformStamped: full pose available
                    pose = PoseData(
                        timestamp=timestamp_us,
                        x=msg.transform.translation.x,
                        y=msg.transform.translation.y,
                        z=msg.transform.translation.z,
                        qw=msg.transform.rotation.w,
                        qx=msg.transform.rotation.x,
                        qy=msg.transform.rotation.y,
                        qz=msg.transform.rotation.z,
                    )
                poses.append(pose)
        
        print(f"Extracted {len(poses)} poses")
        return sorted(poses, key=lambda x: x.timestamp)
    
    def _extract_imu_for_orientation(self) -> Optional[List[Tuple[int, np.ndarray]]]:
        """
        Extract IMU data to estimate orientation for machine_hall dataset.
        
        Returns:
            List of (timestamp_us, orientation_quat) tuples or None
        """
        if self.dataset_type != 'machine_hall':
            return None
        
        print("Extracting IMU for orientation estimation...")
        imu_data = []
        topics = self.TOPICS[self.dataset_type]
        
        # For machine_hall, we'll use IMU to estimate orientation
        # This is a simple integration approach
        dt = 1.0 / 200.0  # IMU rate ~200Hz
        orientation = R.from_quat([0, 0, 0, 1])  # Identity
        
        with rosbag.Bag(self.bag_path, 'r') as bag:
            msg_count = bag.get_message_count(topics['imu'])
            prev_time = None
            
            for topic, msg, t in tqdm(bag.read_messages(topics=[topics['imu']]),
                                       total=msg_count, desc="Processing IMU"):
                timestamp_ns = msg.header.stamp.to_nsec()
                timestamp_us = timestamp_ns // 1000
                
                if prev_time is not None:
                    dt = (timestamp_us - prev_time) / 1e6  # Convert to seconds
                
                # Get angular velocity
                omega = np.array([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ])
                
                # Integrate orientation
                if np.linalg.norm(omega) > 1e-10 and dt > 0:
                    delta_rot = R.from_rotvec(omega * dt)
                    orientation = orientation * delta_rot
                
                quat = orientation.as_quat()  # [x, y, z, w]
                imu_data.append((timestamp_us, quat))
                prev_time = timestamp_us
        
        return imu_data
    
    def _compute_velocities_and_accelerations(
        self, 
        poses: List[PoseData]
    ) -> List[PoseData]:
        """
        Compute velocities and accelerations from position data.
        
        Args:
            poses: List of PoseData with positions
            
        Returns:
            Updated list of PoseData with velocities and accelerations
        """
        print("Computing velocities and accelerations...")
        
        if len(poses) < 3:
            return poses
        
        for i in range(1, len(poses) - 1):
            # Time differences
            dt_prev = (poses[i].timestamp - poses[i-1].timestamp) / 1e6  # seconds
            dt_next = (poses[i+1].timestamp - poses[i].timestamp) / 1e6  # seconds
            
            if dt_prev <= 0 or dt_next <= 0:
                continue
            
            # Central difference for velocity
            vx = (poses[i+1].x - poses[i-1].x) / (dt_prev + dt_next)
            vy = (poses[i+1].y - poses[i-1].y) / (dt_prev + dt_next)
            
            poses[i].vx = vx
            poses[i].vy = vy
        
        # Handle boundaries
        if len(poses) >= 2:
            poses[0].vx = poses[1].vx
            poses[0].vy = poses[1].vy
            poses[-1].vx = poses[-2].vx
            poses[-1].vy = poses[-2].vy
        
        # Compute accelerations
        for i in range(1, len(poses) - 1):
            dt_prev = (poses[i].timestamp - poses[i-1].timestamp) / 1e6
            dt_next = (poses[i+1].timestamp - poses[i].timestamp) / 1e6
            
            if dt_prev <= 0 or dt_next <= 0:
                continue
            
            ax = (poses[i+1].vx - poses[i-1].vx) / (dt_prev + dt_next)
            ay = (poses[i+1].vy - poses[i-1].vy) / (dt_prev + dt_next)
            
            poses[i].ax = ax
            poses[i].ay = ay
        
        # Handle boundaries
        if len(poses) >= 2:
            poses[0].ax = poses[1].ax
            poses[0].ay = poses[1].ay
            poses[-1].ax = poses[-2].ax
            poses[-1].ay = poses[-2].ay
        
        return poses
    
    def _interpolate_poses_to_images(
        self,
        images: List[Tuple[int, str]],
        poses: List[PoseData],
        imu_orientations: Optional[List[Tuple[int, np.ndarray]]] = None
    ) -> Dict[str, PoseData]:
        """
        Interpolate poses to match image timestamps.
        
        Args:
            images: List of (timestamp_us, filename) tuples
            poses: List of PoseData
            imu_orientations: Optional IMU orientation data
            
        Returns:
            Dictionary mapping image filenames to interpolated poses
        """
        print("Interpolating poses to image timestamps...")
        
        if len(poses) < 2:
            raise ValueError("Not enough pose data for interpolation")
        
        # Create interpolation functions for each pose component
        pose_times = np.array([p.timestamp for p in poses])
        
        interp_funcs = {
            'x': interp1d(pose_times, [p.x for p in poses], kind='linear', fill_value='extrapolate'),
            'y': interp1d(pose_times, [p.y for p in poses], kind='linear', fill_value='extrapolate'),
            'z': interp1d(pose_times, [p.z for p in poses], kind='linear', fill_value='extrapolate'),
            'vx': interp1d(pose_times, [p.vx for p in poses], kind='linear', fill_value='extrapolate'),
            'vy': interp1d(pose_times, [p.vy for p in poses], kind='linear', fill_value='extrapolate'),
            'ax': interp1d(pose_times, [p.ax for p in poses], kind='linear', fill_value='extrapolate'),
            'ay': interp1d(pose_times, [p.ay for p in poses], kind='linear', fill_value='extrapolate'),
        }
        
        # For quaternions, we need SLERP or nearest neighbor
        quat_data = np.array([[p.qx, p.qy, p.qz, p.qw] for p in poses])
        
        # If using IMU orientations for machine_hall
        if imu_orientations is not None and self.dataset_type == 'machine_hall':
            imu_times = np.array([t for t, _ in imu_orientations])
            imu_quats = np.array([q for _, q in imu_orientations])
            quat_interp_times = imu_times
            quat_interp_data = imu_quats
        else:
            quat_interp_times = pose_times
            quat_interp_data = quat_data
        
        # Interpolate for each image
        result = {}
        for img_time, img_filename in tqdm(images, desc="Interpolating"):
            # Find nearest quaternion (simple nearest neighbor for orientation)
            quat_idx = np.argmin(np.abs(quat_interp_times - img_time))
            quat = quat_interp_data[quat_idx]
            
            pose = PoseData(
                timestamp=int(img_time),
                x=float(interp_funcs['x'](img_time)),
                y=float(interp_funcs['y'](img_time)),
                z=float(interp_funcs['z'](img_time)),
                qx=float(quat[0]),
                qy=float(quat[1]),
                qz=float(quat[2]),
                qw=float(quat[3]) if len(quat) > 3 else float(quat_interp_data[quat_idx][3]),
                vx=float(interp_funcs['vx'](img_time)),
                vy=float(interp_funcs['vy'](img_time)),
                ax=float(interp_funcs['ax'](img_time)),
                ay=float(interp_funcs['ay'](img_time)),
            )
            
            # Handle quaternion format difference
            if len(quat) == 4:
                pose.qx = float(quat[0])
                pose.qy = float(quat[1])
                pose.qz = float(quat[2])
                pose.qw = float(quat[3])
            
            result[img_filename] = pose
        
        return result
    
    def _downsample_images(
        self, 
        images: List[Tuple[int, str]]
    ) -> List[Tuple[int, str]]:
        """
        Downsample images to target FPS.
        
        Args:
            images: List of (timestamp_us, filename) tuples
            
        Returns:
            Downsampled list
        """
        if self.target_fps is None or len(images) < 2:
            return images
        
        print(f"Downsampling to {self.target_fps} FPS...")
        
        # Calculate original FPS
        total_time = (images[-1][0] - images[0][0]) / 1e6  # seconds
        original_fps = len(images) / total_time
        print(f"Original FPS: {original_fps:.2f}")
        
        if original_fps <= self.target_fps:
            print("Original FPS is already <= target FPS, no downsampling needed")
            return images
        
        # Downsample
        target_interval = 1.0 / self.target_fps * 1e6  # microseconds
        result = [images[0]]
        last_time = images[0][0]
        
        for img_time, img_filename in images[1:]:
            if img_time - last_time >= target_interval:
                result.append((img_time, img_filename))
                last_time = img_time
        
        print(f"Downsampled from {len(images)} to {len(result)} images")
        return result
    
    def _generate_meta_json(
        self,
        images: List[Tuple[int, str]],
        image_poses: Dict[str, PoseData]
    ) -> Tuple[Dict, Dict]:
        """
        Generate meta JSON data.
        
        Args:
            images: List of (timestamp_us, filename) tuples
            image_poses: Dictionary mapping filenames to poses
            
        Returns:
            Tuple of (sequence_meta, ego_meta)
        """
        print("Generating meta JSON...")
        
        # Sequence meta (for test_meta.json)
        sequence_meta = {
            'CAM_F0': [img[1] for img in images],
            'scene': f'{self.scene_name}_scene_0',
            'data_root': f'sensor_blobs/{self.scene_name}',
            'pose': f'ego_meta/{self.scene_name}.json',
        }
        
        # Ego meta (for ego_meta/{scene_name}.json)
        ego_meta = {
            'CAM_F0': {}
        }
        
        for img_time, img_filename in images:
            pose = image_poses.get(img_filename)
            if pose is None:
                continue
            
            key = f'CAM_F0/{img_filename}'
            ego_meta['CAM_F0'][key] = {
                'x': pose.x,
                'y': pose.y,
                'z': pose.z,
                'qw': pose.qw,
                'qx': pose.qx,
                'qy': pose.qy,
                'qz': pose.qz,
                'vx': pose.vx,
                'vy': pose.vy,
                'ax': pose.ax,
                'ay': pose.ay,
                'timestamp': pose.timestamp,
            }
        
        return sequence_meta, ego_meta
    
    def convert(self) -> Dict:
        """
        Run the full conversion process.
        
        Returns:
            Sequence metadata for this scene
        """
        print(f"\n{'='*60}")
        print(f"Converting: {self.scene_name}")
        print(f"Bag path: {self.bag_path}")
        print(f"Output dir: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract images
        images = self._extract_images()
        
        # Step 2: Downsample if needed
        images = self._downsample_images(images)
        
        # Step 3: Extract poses
        poses = self._extract_poses()
        
        # Step 4: Extract IMU for orientation (machine_hall only)
        imu_orientations = self._extract_imu_for_orientation()
        
        # Step 5: Compute velocities and accelerations
        poses = self._compute_velocities_and_accelerations(poses)
        
        # Step 6: Interpolate poses to image timestamps
        image_poses = self._interpolate_poses_to_images(images, poses, imu_orientations)
        
        # Step 7: Generate meta JSON
        sequence_meta, ego_meta = self._generate_meta_json(images, image_poses)
        
        # Step 8: Save ego meta
        ego_meta_path = self.ego_meta_dir / f'{self.scene_name}.json'
        with open(ego_meta_path, 'w') as f:
            json.dump(ego_meta, f)
        print(f"Saved ego meta to: {ego_meta_path}")
        
        print(f"\nConversion complete for {self.scene_name}")
        print(f"  - Images: {len(images)}")
        print(f"  - Poses: {len(image_poses)}")
        
        return sequence_meta


def convert_all_datasets(config_path: str):
    """
    Convert all datasets specified in the config file.
    
    Args:
        config_path: Path to the YAML config file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['output_dir']
    all_sequences = []
    
    # Process each dataset
    for dataset in config['datasets']:
        bag_path = dataset['bag_path']
        scene_name = dataset['scene_name']
        dataset_type = dataset.get('dataset_type', 'auto')
        
        if not os.path.exists(bag_path):
            print(f"Warning: Bag file not found: {bag_path}, skipping...")
            continue
        
        converter = EurocBagConverter(
            bag_path=bag_path,
            output_dir=output_dir,
            scene_name=scene_name,
            dataset_type=dataset_type,
            convert_grayscale_to_rgb=config.get('convert_grayscale_to_rgb', True),
            image_quality=config.get('image_quality', 95),
            target_fps=config.get('target_fps', None),
        )
        
        sequence_meta = converter.convert()
        all_sequences.append(sequence_meta)
    
    # Save combined meta JSON
    meta_json_path = os.path.join(output_dir, 'test_meta.json')
    with open(meta_json_path, 'w') as f:
        json.dump(all_sequences, f, indent=2)
    print(f"\nSaved combined meta to: {meta_json_path}")
    
    print(f"\n{'='*60}")
    print(f"All conversions complete!")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert UAV ROS bags to Epona format'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config', 
        type=str, 
        help='Path to YAML config file for batch conversion'
    )
    group.add_argument(
        '--bag_path',
        type=str,
        help='Path to single ROS bag file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./uav_epona_dataset',
        help='Output directory'
    )
    parser.add_argument(
        '--scene_name',
        type=str,
        help='Scene name (required if using --bag_path)'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=['auto', 'machine_hall', 'vicon_room'],
        default='auto',
        help='Dataset type'
    )
    parser.add_argument(
        '--convert_grayscale_to_rgb',
        action='store_true',
        default=True,
        help='Convert grayscale images to RGB'
    )
    parser.add_argument(
        '--target_fps',
        type=float,
        default=None,
        help='Target FPS for downsampling (None = keep original)'
    )
    parser.add_argument(
        '--image_quality',
        type=int,
        default=95,
        help='JPEG quality (1-100)'
    )
    
    args = parser.parse_args()
    
    if args.config:
        convert_all_datasets(args.config)
    else:
        if not args.scene_name:
            parser.error("--scene_name is required when using --bag_path")
        
        converter = EurocBagConverter(
            bag_path=args.bag_path,
            output_dir=args.output_dir,
            scene_name=args.scene_name,
            dataset_type=args.dataset_type,
            convert_grayscale_to_rgb=args.convert_grayscale_to_rgb,
            image_quality=args.image_quality,
            target_fps=args.target_fps,
        )
        
        sequence_meta = converter.convert()
        
        # Save single sequence meta
        meta_json_path = os.path.join(args.output_dir, 'test_meta.json')
        with open(meta_json_path, 'w') as f:
            json.dump([sequence_meta], f, indent=2)
        print(f"Saved meta to: {meta_json_path}")


if __name__ == '__main__':
    main()

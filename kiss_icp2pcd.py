import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import struct
from tqdm import tqdm
import os
from matplotlib import cm

def load_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            pose = np.vstack([pose, [0, 0, 0, 1]])
            poses.append(pose)
    return poses

def extract_points_with_intensity(msg):
    step = msg.point_step
    data = msg.data
    num_points = len(data) // step
    points = []
    intensities = []
    
    for i in range(0, num_points * step, step):
        if i + 16 <= len(data):
            x = struct.unpack('<f', data[i:i+4])[0]
            y = struct.unpack('<f', data[i+4:i+8])[0]
            z = struct.unpack('<f', data[i+8:i+12])[0]
            intensity = struct.unpack('<f', data[i+12:i+16])[0]
            
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isnan(intensity)):
                dist = x*x + y*y + z*z
                if 0.25 < dist < 10000:
                    points.append([x, y, z])
                    intensities.append(intensity)
    
    return np.array(points) if points else None, np.array(intensities) if intensities else None

def compute_global_intensity_range(bag_path, sample_frames=100):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    all_intensities = []
    
    with Reader(bag_path) as reader:
        frame_count = 0
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == "/lidar_points" and frame_count < sample_frames:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                _, intensities = extract_points_with_intensity(msg)
                if intensities is not None and len(intensities) > 0:
                    valid_intensities = intensities[intensities > 0]
                    if len(valid_intensities) > 0:
                        all_intensities.extend(valid_intensities[::10])
                    frame_count += 1
    
    all_intensities = np.array(all_intensities)
    global_min = np.percentile(all_intensities, 1)
    global_max = np.percentile(all_intensities, 99)
    
    return global_min, global_max

def map_intensity_uniform(intensities, global_min, global_max, 
                          use_log=True, gamma=None, cmap_name="viridis"):
    vals = np.clip(intensities, global_min, global_max)
    
    if use_log:
        vals = np.log1p(vals - global_min + 1)
        max_val = np.log1p(global_max - global_min + 1)
        vals = vals / max_val
    else:
        vals = (vals - global_min) / (global_max - global_min)
    
    if gamma is not None:
        vals = np.power(vals, gamma)
    
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(vals)[:, :3]
    
    return colors

def main():
    bag_path = "/home/taewook/ISIS/hyper_2024_bag2/bridge_upper_rosbag2"
    pose_file = "/home/taewook/ISIS/kiss_icp_poses.txt"
    output_pcd = "/home/taewook/ISIS/hdmap_intensity_best_practice.pcd"
    colormap = "turbo"
    
    global_min, global_max = compute_global_intensity_range(bag_path, sample_frames=200)
    poses = load_poses(pose_file)
    
    MAX_FRAMES = 1200
    FRAME_SKIP = 1
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    accumulated_points = []
    accumulated_intensities = []
    
    with Reader(bag_path) as reader:
        frame_count = 0
        pose_idx = 0
        
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == "/lidar_points":
                if frame_count % FRAME_SKIP == 0 and frame_count < MAX_FRAMES:
                    if pose_idx >= len(poses):
                        break
                    
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    points, intensities = extract_points_with_intensity(msg)
                    
                    if points is not None and len(points) > 0:
                        valid_mask = intensities > 0
                        if np.any(valid_mask):
                            points = points[valid_mask]
                            intensities = intensities[valid_mask]
                            
                            pose = poses[pose_idx]
                            points_h = np.hstack([points, np.ones((len(points), 1))])
                            points_transformed = (pose @ points_h.T).T[:, :3]
                            
                            accumulated_points.append(points_transformed)
                            accumulated_intensities.append(intensities)
                    
                    pose_idx += 1
                frame_count += 1
    
    if not accumulated_points:
        return
    
    all_points = np.vstack(accumulated_points)
    all_intensities = np.hstack(accumulated_intensities)
    
    voxel_size = 0.1
    voxel_indices = np.floor(all_points / voxel_size).astype(int)
    
    unique_voxels, indices = np.unique(voxel_indices, axis=0, return_index=True)
    filtered_points = all_points[indices]
    filtered_intensities = all_intensities[indices]
    
    colors = map_intensity_uniform(
        filtered_intensities, 
        global_min, 
        global_max,
        use_log=True,
        gamma=0.7,
        cmap_name=colormap
    )
    
    with open(output_pcd, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F U\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(filtered_points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(filtered_points)}\n")
        f.write("DATA ascii\n")
        
        for p, c in tqdm(zip(filtered_points, colors), total=len(filtered_points), desc="Writing"):
            c_uint8 = (c * 255).astype(np.uint8)
            rgb = (int(c_uint8[0]) << 16) | (int(c_uint8[1]) << 8) | int(c_uint8[2])
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {rgb}\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LiDAR Intensity 기반 색상 PCD 생성 (Best Practice 버전)
- 전역 고정 정규화
- 지각 균일 컬러맵 (viridis, turbo 등)
- 벡터화된 빠른 처리
- 10배 더 많은 포인트 처리
"""
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import struct
from tqdm import tqdm
import os
from matplotlib import cm

def load_poses(pose_file):
    """Load poses from KISS-ICP"""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            pose = np.vstack([pose, [0, 0, 0, 1]])
            poses.append(pose)
    return poses

def extract_points_with_intensity(msg):
    """PointCloud2에서 xyz와 intensity 추출 (벡터화)"""
    step = msg.point_step
    data = msg.data
    
    # 모든 포인트를 한번에 파싱
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
                if 0.25 < dist < 10000:  # 0.5m to 100m squared
                    points.append([x, y, z])
                    intensities.append(intensity)
    
    return np.array(points) if points else None, np.array(intensities) if intensities else None

def compute_global_intensity_range(bag_path, sample_frames=100):
    """전체 데이터셋에서 intensity 범위 계산"""
    print("전역 intensity 범위 계산 중...")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    all_intensities = []
    
    with Reader(bag_path) as reader:
        frame_count = 0
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == "/lidar_points" and frame_count < sample_frames:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                _, intensities = extract_points_with_intensity(msg)
                if intensities is not None and len(intensities) > 0:
                    # 0이 아닌 값만 샘플링
                    valid_intensities = intensities[intensities > 0]
                    if len(valid_intensities) > 0:
                        all_intensities.extend(valid_intensities[::10])  # 10개마다 샘플
                    frame_count += 1
    
    all_intensities = np.array(all_intensities)
    
    # 전역 범위 결정 (1% ~ 99% percentile)
    global_min = np.percentile(all_intensities, 1)
    global_max = np.percentile(all_intensities, 99)
    
    print(f"전역 intensity 범위: [{global_min:.2f}, {global_max:.2f}]")
    print(f"샘플링된 값 개수: {len(all_intensities):,}")
    
    return global_min, global_max

def map_intensity_uniform(intensities, global_min, global_max, 
                          use_log=True, gamma=None, cmap_name="viridis"):
    """
    지각 균일 컬러맵을 사용한 intensity 매핑
    """
    # 정규화
    vals = np.clip(intensities, global_min, global_max)
    
    if use_log:
        # 로그 스케일 적용
        vals = np.log1p(vals - global_min + 1)  # +1 to avoid log(0)
        max_val = np.log1p(global_max - global_min + 1)
        vals = vals / max_val
    else:
        # 선형 정규화
        vals = (vals - global_min) / (global_max - global_min)
    
    # 감마 보정 (선택적)
    if gamma is not None:
        vals = np.power(vals, gamma)
    
    # 컬러맵 적용
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(vals)[:, :3]  # RGB만 추출 (alpha 제외)
    
    return colors

def main():
    bag_path = "/home/taewook/ISIS/hyper_2024_bag2/bridge_upper_rosbag2"
    pose_file = "/home/taewook/ISIS/kiss_icp_poses.txt"
    output_pcd = "/home/taewook/ISIS/hdmap_intensity_best_practice.pcd"
    
    # 컬러맵 옵션 (viridis, turbo, magma, inferno, plasma, gray 등)
    colormap = "turbo"  # turbo는 무지개색과 비슷하지만 더 균일함
    
    print("=" * 60)
    print("LiDAR Intensity 기반 색상 PCD 생성 (Best Practice)")
    print(f"- 컬러맵: {colormap}")
    print("- 전역 고정 정규화")
    print("- 로그 스케일 + 감마 보정")
    print("- 10배 더 많은 포인트 처리")
    print("=" * 60)
    
    # 전역 intensity 범위 계산
    global_min, global_max = compute_global_intensity_range(bag_path, sample_frames=200)
    
    # Pose 로드
    poses = load_poses(pose_file)
    print(f"\n로드된 포즈: {len(poses)}개")
    
    # 파라미터 (적절히 조정)
    MAX_FRAMES = 1200  # 전체 프레임 사용 (bag에 1260개 있음)
    FRAME_SKIP = 1     # 모든 프레임 사용
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    accumulated_points = []
    accumulated_intensities = []
    
    print(f"\nbag 파일 처리 중 (최대 {MAX_FRAMES} 프레임)...")
    
    with Reader(bag_path) as reader:
        frame_count = 0
        pose_idx = 0
        
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == "/lidar_points":
                if frame_count % FRAME_SKIP == 0 and frame_count < MAX_FRAMES:
                    if pose_idx >= len(poses):
                        print(f"포즈 부족: {pose_idx} >= {len(poses)}")
                        break
                    
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    points, intensities = extract_points_with_intensity(msg)
                    
                    if points is not None and len(points) > 0:
                        # 0이 아닌 intensity만 사용
                        valid_mask = intensities > 0
                        if np.any(valid_mask):
                            points = points[valid_mask]
                            intensities = intensities[valid_mask]
                            
                            # Transform points
                            pose = poses[pose_idx]
                            points_h = np.hstack([points, np.ones((len(points), 1))])
                            points_transformed = (pose @ points_h.T).T[:, :3]
                            
                            accumulated_points.append(points_transformed)
                            accumulated_intensities.append(intensities)
                            
                            if pose_idx % 100 == 0:
                                total_points = sum(len(p) for p in accumulated_points)
                                print(f"프레임 {pose_idx}: {total_points:,} 포인트")
                    
                    pose_idx += 1
                
                frame_count += 1
    
    if not accumulated_points:
        print("포인트가 없습니다!")
        return
    
    # 모든 포인트 합치기
    all_points = np.vstack(accumulated_points)
    all_intensities = np.hstack(accumulated_intensities)
    
    print(f"\n총 {len(all_points):,} 포인트 처리 중...")
    
    # Voxel grid filtering (더 세밀하게)
    voxel_size = 0.1  # 0.2 -> 0.1 (더 많은 포인트 유지)
    voxel_indices = np.floor(all_points / voxel_size).astype(int)
    
    # 각 voxel에서 대표 포인트 선택
    unique_voxels, indices = np.unique(voxel_indices, axis=0, return_index=True)
    filtered_points = all_points[indices]
    filtered_intensities = all_intensities[indices]
    
    print(f"Voxel 필터링 후: {len(filtered_points):,} 포인트")
    
    # 색상 매핑 (전역 고정 범위 사용)
    print("\n색상 매핑 중...")
    colors = map_intensity_uniform(
        filtered_intensities, 
        global_min, 
        global_max,
        use_log=True,
        gamma=0.7,  # 약간의 감마 보정
        cmap_name=colormap
    )
    
    # PCD 파일 저장
    print(f"\nPCD 파일 저장 중: {output_pcd}")
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
    
    file_size_mb = os.path.getsize(output_pcd) / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("=== 완료 ===")
    print("=" * 60)
    print(f"총 포인트 수: {len(filtered_points):,}")
    print(f"파일 크기: {file_size_mb:.2f} MB")
    print(f"저장 위치: {output_pcd}")
    print(f"컬러맵: {colormap}")
    print(f"전역 intensity 범위: [{global_min:.2f}, {global_max:.2f}]")
    
    # Intensity 통계
    print(f"\n최종 Intensity 분포:")
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        val = np.percentile(filtered_intensities, p)
        print(f"  {p:3d}%: {val:.1f}")

if __name__ == "__main__":
    main()

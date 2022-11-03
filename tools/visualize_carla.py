import argparse
import pickle

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0, required=False)
    args = parser.parse_args()

    with open("data/carla_bench/c16h35_bench/carla_infos_train.pkl", "rb") as f:
        data_infos = pickle.load(f)

    data_info = data_infos[args.id]

    scene_id = data_info["scene_id"]

    points = np.fromfile(f"data/carla_bench/c16h35_bench/velodyne/{scene_id}", dtype=np.float32)
    points = points.reshape(-1, 4)
    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])

    bboxes = []
    for bbox in data_info["annos"]["bboxes_3d"]:
        bbox = np.asarray(bbox)
        bottom_center = bbox[:3]
        extent = bbox[3:6]
        yaw = bbox[6]

        center = bottom_center.copy()
        center[2] += extent[2] / 2

        r = R.from_euler("z", yaw, degrees=False)

        bbox = o3d.geometry.OrientedBoundingBox(
            center=center,
            R=r.as_matrix(),
            extent=extent,
        )
        bbox.color = [1, 0, 0]
        bboxes.append(bbox)

    o3d.visualization.draw([pc] + bboxes)


if __name__ == "__main__":
    main()

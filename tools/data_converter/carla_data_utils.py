from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import yaml
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


@dataclass
class Pose:
    x: float
    y: float
    z: float
    # degree
    pitch: float
    yaw: float
    roll: float

    @classmethod
    def from_list(cls, args: List[float]) -> "Pose":
        assert len(args) == 6
        return cls(*args)

    def get_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        r = R.from_euler(
            "zyx", (self.yaw, self.pitch, self.roll), degrees=True
        )
        t = np.asarray([self.x, self.y, self.z])

        return r.as_matrix(), t


@dataclass
class LidarInfo:
    index: int
    sensor_rot: np.ndarray
    sensor_trans: np.ndarray
    pc_path: str


@dataclass
class VehicleInfo:
    id: int
    rotation: List[float]      # pitch, yaw, roll
    center: List[float]
    extent: List[float]
    location: List[float]

    def get_bbox(self) -> List[float]:
        center = np.asarray(self.location) + np.asarray(self.center)
        extent = np.asarray(self.extent) * 2

        bottom_center = center.copy()
        bottom_center[2] -= extent[2] / 2
        bottom_center[1] = -bottom_center[1]

        return bottom_center.tolist() + extent.tolist() + [
            np.deg2rad(self.rotation[1])
        ]


@dataclass
class DataInfo:
    scene_id: str
    lidars: List[LidarInfo]
    vehicles: List[VehicleInfo]


def load_raw_data_infos(data_path: Path) -> List[DataInfo]:
    scene_ids = set([
        str(anno_path.name).split('_')[0]
        for anno_path in data_path.glob("*.yaml")             
    ])

    data_infos = []
    
    for scene_id in tqdm(scene_ids):
        lidar_infos = []
        vehicle_infos = {}
        valid = True

        for i in range(4):
            anno_path = data_path / f"{scene_id}_lidar{i}.yaml"
            if not anno_path.exists():
                continue
 

            with open(anno_path, "r") as f:
                anno = yaml.load(f, yaml.SafeLoader)

            lidar_pose = Pose.from_list(anno["lidar_pose"])
            sensor_rot, sensor_trans = lidar_pose.get_transform()
            pc_path = anno_path.stem + ".pcd"
            if not (data_path / pc_path).exists():
                valid = False
                break

            lidar_infos.append(
                LidarInfo(
                    index=i,
                    sensor_rot=sensor_rot,
                    sensor_trans=sensor_trans,
                    pc_path=anno_path.stem + ".pcd",
                )
            )

            vehicle_infos.update({
                v_id: VehicleInfo(
                    id=v_id,
                    rotation=info["rotation"],
                    center=info["center"],
                    extent=info["extent"],
                    location=info["location"],
                )
                for v_id, info in anno["vehicles"].items()
            })

        if not valid:
            continue

        vehicle_infos = list(vehicle_infos.values())

        data_infos.append(
            DataInfo(
                scene_id=scene_id,
                lidars=lidar_infos,
                vehicles=vehicle_infos
            )
        )

    return data_infos

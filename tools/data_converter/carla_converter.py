from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import pickle
import random

import yaml
import numpy as np
import mmcv
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .carla_data_utils import DataInfo


class CarlaConverter:
    def __init__(
        self,
        data_path: Path,
        out_path: Path,
        raw_data_infos: List[DataInfo],
        num_workers: int,
    ):
        self.data_path = data_path
        self.out_path = out_path
        self.raw_data_infos = raw_data_infos
        self.num_workers = num_workers

    def convert(self):
        print('Start converting ...')
        data_infos = mmcv.track_parallel_progress(
            self.convert_one,
            tasks=self.raw_data_infos,
            nproc=self.num_workers,
        )

        r = random.Random()
        r.seed(233)
        r.shuffle(data_infos)
        random.shuffle(data_infos)

        num_train_infos = int(len(data_infos) * 0.8)
        train_data_infos = data_infos[:num_train_infos]
        val_data_infos = data_infos[num_train_infos:]

        with open(self.out_path / "carla_infos_train.pkl", "wb") as f:
            pickle.dump(train_data_infos, f)

        with open(self.out_path / "carla_infos_val.pkl", "wb") as f:
            pickle.dump(val_data_infos, f)

        print('\nFinished ...')

    def convert_one(self, raw_data_info: DataInfo):
        points_list = []
        for lidar_info in raw_data_info.lidars:
            pc = o3d.io.read_point_cloud(
                str(self.data_path / "raw_data" / lidar_info.pc_path)
            )
            pc.rotate(lidar_info.sensor_rot)
            pc.translate(lidar_info.sensor_trans)

            points = np.asarray(pc.points).astype(np.float32)
            colors = np.asarray(pc.colors).astype(np.float32)
            points_list.append(
                np.concatenate([points, colors], axis=-1)
            )

        points = np.concatenate(points_list, axis=0)
        points.tofile(self.data_path / "velodyne" / raw_data_info.scene_id)

        gt_bboxes_3d = []
        gt_names = []
        difficulty = []
        for vehicle in raw_data_info.vehicles:
            bbox = vehicle.get_bbox()
            assert len(bbox) == 7
            gt_bboxes_3d.append(bbox)
            gt_names.append("Car")
            difficulty.append(0)

        data_info = {
            "scene_id": raw_data_info.scene_id,
            "annos": {
                "bboxes_3d": gt_bboxes_3d,
                "name": gt_names,
                "difficulty": difficulty,
            },
        }

        return data_info


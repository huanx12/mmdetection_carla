from pathlib import Path
from os import path as osp
import copy

import numpy as np
import mmcv
import torch

from ..core.bbox.structures import (
    LiDARInstance3DBoxes, points_cam2img, Box3DMode
)
from .builder import DATASETS
from .kitti_dataset import KittiDataset


@DATASETS.register_module()
class CarlaDataset(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Carla Dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to "velodyne".
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to "LiDAR" in this dataset. Available options includes

            - "LiDAR": box in LiDAR coordinates
            - "Depth": box in depth coordinates, usually for indoor dataset
            - "Camera": box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list(float), optional): The range of point cloud used
            to filter invalid predicted boxes.
            Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ("Car",)

    def __init__(
        self,
        data_root,
        ann_file,
        split,
        pts_prefix = "velodyne",
        pipeline = None,
        classes = None,
        modality = None,
        box_type_3d = "LiDAR",
        filter_empty_gt = True,
        test_mode = False,
        load_interval = 1,
        pcd_limit_range = [-85, -85, -5, 85, 85, 5],
        **kwargs
    ):
        assert box_type_3d == "LiDAR"

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range,
            **kwargs,
        )

        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, "flag"):
            self.flag = self.flag[::load_interval]

    def _get_pts_filename(self, idx):
        """Get point cloud filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        scene_id = self.data_infos[idx]["scene_id"]
        pts_filename = osp.join(self.data_root, self.pts_prefix, scene_id)
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        scene_id = info["scene_id"]

        pts_filename = self._get_pts_filename(index)
        input_dict = dict(
            sample_idx=scene_id,
            pts_filename=pts_filename,
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                    0, 1, 2 represent xxxxx respectively.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        # difficulty = info["annos"]["difficulty"]
        annos = info["annos"]
        gt_bboxes_3d = np.asarray(annos["bboxes_3d"])
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)
        gt_names = annos["name"]
        difficulty = annos["difficulty"]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.asarray(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            labels=gt_labels,
            gt_names=gt_names,
            difficulty=difficulty
        )
        return anns_results

    def bbox2result_kitti(
        self,
        net_outputs,
        class_names,
        pklfile_prefix=None,
        submission_prefix=None,
    ):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            "invalid list length of network outputs"
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print("\nConverting prediction to KITTI format")
        for idx, pred_dicts in enumerate(
            mmcv.track_iter_progress(net_outputs)
        ):
            annos = []
            info = self.data_infos[idx]
            scene_id = info["scene_id"]
            # image_shape = info["image_shape"][:2]
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                "name": [],
                "truncated": [],
                "occluded": [],
                "alpha": [],
                "bbox": [],
                "dimensions": [],
                "location": [],
                "rotation_y": [],
                "score": []
            }
            if len(box_dict["bbox"]) > 0:
                # box_2d_preds = box_dict["bbox"]
                box_preds = box_dict["box3d_camera"]
                scores = box_dict["scores"]
                box_preds_lidar = box_dict["box3d_lidar"]
                label_preds = box_dict["label_preds"]

                for box, box_lidar, score, label in zip(
                    box_preds, box_preds_lidar, scores, label_preds
                ):
                    # bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    # bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno["name"].append(class_names[int(label)])
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["alpha"].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6]
                    )
                    # anno["bbox"].append(bbox)
                    anno["bbox"].append([0, 0, 100, 100])
                    anno["dimensions"].append(box[3:6])
                    anno["location"].append(box[:3])
                    anno["rotation_y"].append(box[6])
                    anno["score"].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    "name": np.array([]),
                    "truncated": np.array([]),
                    "occluded": np.array([]),
                    "alpha": np.array([]),
                    "bbox": np.zeros([0, 4]),
                    "dimensions": np.zeros([0, 3]),
                    "location": np.zeros([0, 3]),
                    "rotation_y": np.array([]),
                    "score": np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f"{submission_prefix}/{scene_id}.txt"
                with open(curr_file, "w") as f:
                    # bbox = anno["bbox"]
                    loc = anno["location"]
                    dims = anno["dimensions"]  # lhw -> hwl

                    for idx in range(len(loc)):
                        print(
                            "{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} "
                            "{:.4f} {:.4f} {:.4f} "
                            "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
                                anno["name"][idx], anno["alpha"][idx],
                                0, 0, 0, 0,
                                dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno["rotation_y"][idx],
                                anno["score"][idx],
                            ),
                            file=f,
                        )

            annos[-1]["scene_id"] = [scene_id] * len(annos[-1]["score"])

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith((".pkl", ".pickle")):
                out = f"{pklfile_prefix}.pkl"
            mmcv.dump(det_annos, out)
            print(f"Result is saved to {out}.")

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict["boxes_3d"]
        scores = box_dict["scores_3d"]
        labels = box_dict["labels_3d"]
        scene_id = info["scene_id"]
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                scene_id=scene_id,
            )

        # rect = info["calib"]["R0_rect"].astype(np.float32)
        # Trv2c = info["calib"]["Tr_velo_to_cam"].astype(np.float32)
        # P2 = info["calib"]["P2"].astype(np.float32)
        # img_shape = info["image_shape"]
        # P2 = box_preds.tensor.new_tensor(P2)

        # box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)
        box_preds_camera = box_preds.convert_to(Box3DMode.CAM)

        # box_corners = box_preds_camera.corners
        # box_corners_in_image = points_cam2img(box_corners, P2)
        # # box_corners_in_image: [N, 8, 2]
        # minxy = torch.min(box_corners_in_image, dim=1)[0]
        # maxxy = torch.max(box_corners_in_image, dim=1)[0]
        # box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # # Post-processing
        # # check box_preds_camera
        # image_shape = box_preds.tensor.new_tensor(img_shape)
        # valid_cam_inds = (
        #     (box_2d_preds[:, 0] < image_shape[1]) &
        #     (box_2d_preds[:, 1] < image_shape[0]) &
        #     (box_2d_preds[:, 2] > 0) &
        #     (box_2d_preds[:, 3] > 0)
        # )
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = (
            (box_preds.center > limit_range[:3]) &
            (box_preds.center < limit_range[3:])
        )
        # valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                # bbox=box_2d_preds[valid_inds, :].numpy(),
                bbox=np.zeros([0, 4]),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                scene_id=scene_id,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                scene_id=scene_id,
            )

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import kitti_eval

        def to_kitti_anno(anno):
            name = anno["name"]
            gt_bboxes_3d = np.asarray(anno["bboxes_3d"])
            gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)
            box_lidar = gt_bboxes_3d.tensor.numpy()
            box = gt_bboxes_3d.convert_to(Box3DMode.CAM).tensor.numpy()

            alpha = -np.arctan2(-box_lidar[:, 1], box_lidar[:, 0]) + box[:, 6]
            alpha = -10
            dimensions = box[:, 3:6]
            location = box[:, :3]
            rotation_y = box[:, 6]

            return {
                "name": name,
                "alpha": alpha,
                "dimensions": dimensions,
                "location": location,
                "rotation_y": rotation_y,
                "bbox": np.asarray([[0, 0, 100, 100] for _ in range(len(name))]),
                "truncated": np.asarray([0.0 for _ in range(len(name))]),
                "occluded": np.asarray([0 for _ in range(len(name))], dtype=np.int64),
            }

        gt_annos = [to_kitti_anno(info["annos"]) for info in self.data_infos]

        if isinstance(result_files, dict):
            ap_dict = {}
            for name, result_files_ in result_files.items():
                eval_types = ["bev", "3d"]
                ap_result_str, ap_dict_ = kitti_eval(
                    gt_annos,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types,
                )
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                mmcv.utils.print_log(
                    f"Results of {name}:\n" + ap_result_str, logger=logger
                )

        else:
            ap_result_str, ap_dict = kitti_eval(
                gt_annos, result_files, self.CLASSES, eval_types=["bev", "3d"]
            )
            mmcv.utils.print_log("\n" + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

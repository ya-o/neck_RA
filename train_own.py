from mmcv import Config
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.apis import train_model
import mmcv

import json
import os.path as osp
from collections import OrderedDict
import tempfile

import numpy as np

from mmpose.core.evaluation.top_down_eval import (keypoint_nme,
                                                  keypoint_pck_accuracy, keypoint_auc)
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class TopDownCOCOTinyDataset(Kpt2dSviewRgbImgTopDownDataset):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info,
            coco_style=False,
            test_mode=test_mode)

        # flip_pairs, upper_body_ids and lower_body_ids will be used
        # in some data augmentations like random flip
        self.ann_info['flip_pairs'] = None
        self.ann_info['upper_body_ids'] = None
        self.ann_info['lower_body_ids'] = None

        self.ann_info['joint_weights'] = None
        self.ann_info['use_different_joint_weights'] = False

        self.dataset_name = 'coco_tiny'
        self.db = self._get_db()

    def _get_db(self):
        with open(self.ann_file) as f:
            anns = json.load(f)

        db = []
        for idx, ann in enumerate(anns):
            # get image path
            image_file = osp.join(self.img_prefix, ann['image_file'])
            # get bbox
            bbox = ann['bbox']
            # get keypoints
            keypoints = np.array(
                ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            num_joints = keypoints.shape[0]
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            sample = {
                'image_file': image_file,
                'bbox': bbox,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_score': 1,
                'bbox_id': idx,
            }
            db.append(sample)

        return db

    def evaluate(self, results, res_folder=None, metric='PCK', **kwargs):
        """Evaluate keypoint detection results. The pose prediction results will
        be saved in `${res_folder}/result_keypoints.json`.

        Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

        Args:
        results (list(preds, boxes, image_path, output_heatmap))
            :preds (np.ndarray[N,K,3]): The first two dimensions are
                coordinates, score is the third dimension of the array.
            :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                , scale[1],area, score]
            :image_paths (list[str]): For example, ['Test/source/0.jpg']
            :output_heatmap (np.ndarray[N, K, H, W]): model outputs.

        res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
        metric (str | list[str]): Metric to be performed.
            Options: 'PCK', 'NME'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'NME', "AUC"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    def _report_metric(self, res_file, metrics, pck_thr=0.5):
        """Keypoint evaluation.

        Args:
        res_file (str): Json file stored prediction results.
        metrics (str | list[str]): Metric to be performed.
            Options: 'PCK', 'NME'.
        pck_thr (float): PCK threshold, default: 0.3.

        Returns:
        dict: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        masks = []

        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)

        normalize_factor = self._get_normalize_factor(gts)

        if 'PCK' in metrics:
            acc, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr, normalize_factor)
            # print("*************")
            # print(pck)
            info_str.append(('PCK', pck))

        if 'NME' in metrics:
            info_str.append(
                ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

        if "AUC" in metrics:
            info_str.append(
                ("AUC", keypoint_auc(outputs, gts, masks, normalize_factor)))

        return info_str

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    @staticmethod
    def _sort_and_unique_bboxes(kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts

    @staticmethod
    def _get_normalize_factor(gts):
        """Get inter-ocular distance as the normalize factor, measured as the
        Euclidean distance between the outer corners of the eyes.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Return:
            np.ndarray[N, 2]: normalized factor
        """

        interocular = np.linalg.norm(
            gts[:, 0, :] - gts[:, 1, :], axis=1, keepdims=True)
        return np.tile(interocular, [1, 2])


cfg = Config.fromfile(
    './configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
)

# set basic configs

cfg.data_root = 'data/NECK'
cfg.work_dir = 'work_dirs/hrnet_w32_coco_tiny_256x192'
cfg.gpu_ids = range(1)
cfg.seed = 0

# set log interval
cfg.log_config.interval = 1

# set evaluation configs
cfg.evaluation.interval = 1
cfg.evaluation.metric = ['PCK', 'NME', "AUC"]
cfg.evaluation.save_best = 'PCK'

# set learning rate policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[17, 35])
cfg.total_epochs = 40

# set batch size
cfg.data.samples_per_gpu = 1
cfg.data.val_dataloader = dict(samples_per_gpu=1)
cfg.data.test_dataloader = dict(samples_per_gpu=1)

# set dataset configs
cfg.data.train.type = 'TopDownCOCOTinyDataset'
cfg.data.train.ann_file = f'{cfg.data_root}/train.json'
cfg.data.train.img_prefix = f'{cfg.data_root}/images/'

cfg.data.val.type = 'TopDownCOCOTinyDataset'
cfg.data.val.ann_file = f'{cfg.data_root}/val.json'
cfg.data.val.img_prefix = f'{cfg.data_root}/images/'

cfg.data.test.type = 'TopDownCOCOTinyDataset'
cfg.data.test.ann_file = f'{cfg.data_root}/val.json'
cfg.data.test.img_prefix = f'{cfg.data_root}/images/'

# print(cfg.pretty_text)


# build dataset
datasets = [build_dataset(cfg.data.train)]

# build model
model = build_posenet(cfg.model)

# create work_dir
mmcv.mkdir_or_exist(cfg.work_dir)

# train model
train_model(
    model, datasets, cfg, distributed=False, validate=True, meta=dict())
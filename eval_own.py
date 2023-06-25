import os.path as osp
from collections import OrderedDict
import tempfile
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmpose.core.evaluation.top_down_eval import (keypoint_nme,
                                                  keypoint_pck_accuracy, keypoint_auc)
import json


class TopDownCOCOTinyDataset():
    def __init__(self, ann_file, img_prefix):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
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
            :boxe_id

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
            bbox_ids = result['bbox_ids']

            kpts.append({
                    'keypoints': preds,
                    'bbox_id': bbox_ids
                })

        kpts = self._sort_and_unique_bboxes(kpts)

        info_str = self._report_metric(kpts, metrics)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    def _report_metric(self, preds, metrics, pck_thr=0.5):
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


img_dir = "data/NECK/images/"
file_name = "data/NECK/train.json"

Dataset = TopDownCOCOTinyDataset(ann_file=file_name, img_prefix=img_dir)

with open("data/NECK/train.json", "r", encoding="utf-8") as f:
    content = json.load(f)

pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
pose_checkpoint = 'work_dirs/hrnet_w32_coco_tiny_256x192/latest.pth'
# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint)

results_ = []
for i in range(len(content)):
    selected = content[i]
    img = img_dir + selected["image_file"]
    bbox = selected["bbox"]
    person = {}
    person['bbox'] = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]), 0.99]
    person_results = [person]
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img,
        person_results,
        bbox_thr=0.3,
        format='xyxy',
        dataset=pose_model.cfg.data.test.type)
    results_.append({'preds': pose_results[0]['keypoints'], 'bbox_ids': i})

out = Dataset.evaluate(results_, metric=['PCK', 'NME', "AUC"])
print(out)

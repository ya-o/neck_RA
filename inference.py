from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np


img_dir = "data/NECK/images/"
index = 0
with open("data/NECK/val.json", "r", encoding="utf-8") as f:
    content = json.load(f)
selected = content[index]

img = img_dir + selected["image_file"]
key_points = selected["keypoints"]
bbox = selected["bbox"]
person = {}
person['bbox'] = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]), 0.99]
person_results = [person]

local_runtime = False

pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
pose_checkpoint = 'work_dirs/hrnet_w32_coco_tiny_256x192/latest.pth'
# det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
# det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint)
# initialize detector
# det_model = init_detector(det_config, det_checkpoint)

# # inference detection
# mmdet_results = inference_detector(det_model, img)

# extract person (COCO_ID=1) bounding boxes from the detection results
# person_results = process_mmdet_results(mmdet_results, cat_id=1)
#
# print(person_results)

# inference pose
pose_results, returned_outputs = inference_top_down_pose_model(
    pose_model,
    img,
    person_results,
    bbox_thr=0.3,
    format='xyxy',
    dataset=pose_model.cfg.data.test.type)

print(pose_results)

# show pose estimation results
vis_result = vis_pose_result(
    pose_model,
    img,
    pose_results,
    radius=6,
    dataset=pose_model.cfg.data.test.type,
    show=False,
    kpt_score_thr=0)

# reduce image size

plt.figure()
plt.title("Pose Estimation")
plt.imshow(vis_result)
plt.show()
plt.close()


# show truth
img2 = cv2.imread(img)

point_r = []
for i in range(4):
    start_point = i * 3
    point_r.append([int(key_points[start_point]), int(key_points[start_point + 1]), 0.99])
point_r = np.array(point_r)

point_info = [{"bbox": np.array(bbox), "keypoints": point_r}]
vis_result2 = vis_pose_result(
    pose_model,
    img,
    point_info,
    radius=6,
    dataset=pose_model.cfg.data.test.type,
    show=False,
    kpt_score_thr=0)

plt.figure()
plt.title("Truth")
plt.imshow(vis_result2)
plt.show()



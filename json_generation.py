import json
import pandas as pd
import cv2


def read_csv(root):
    data_all = pd.read_csv(root)
    data = data_all[["Pic", "ADI_L_X", "ADI_L_Y", "ADI_R_X", "ADI_R_Y", "SAC_L_X", "SAC_L_Y", "SAC_R_X", "SAC_R_Y"]]
    data = data.values
    return data


def split_list_n_list(original_list, n):
    # spilt data by folder
    cnt = len(original_list) // n
    back = []
    for i in range(0, n):
        if i == n-1:
            back.append(original_list[i*cnt:])
        else:
            back.append(original_list[i*cnt:(i+1)*cnt])
    return back


def make_json(data, mode):
    record = []
    for i in range(len(data)):
        current_data = data[i]
        pic_id = current_data[0]
        img_name = "neck" + str(pic_id) + ".jpg"
        img_root = img_dir + img_name
        img = cv2.imread(img_root)
        w = img.shape[1]
        h = img.shape[0]
        img_size = [w, h]
        box = [2, 2, w - 10, h - 10]
        key_point = []
        add = [2, 4, 6, 8]

        for j in range(1, 9):
            key_point.append(int(current_data[j]))
            if j in add:
                key_point.append(1)

        record.append({"image_file": img_name, "image_size": img_size, "bbox": box, "keypoints": key_point})

    json_str = json.dumps(record)
    with open(f'data/NECK/{mode}.json', 'w') as json_file:
        json_file.write(json_str)


split = [[0], [1, 2], [3, 4], [5], [6], [7, 8], [9], [10], [11], [12, 13]]
index_ = 0

img_dir = "data/NECK/images/"
point_data = list(read_csv("annotation.csv"))

select = split[index_]
val_list = []
for i in range(len(select)):
    id = select[i] - i
    out = point_data.pop(id)
    val_list.append(out)
print(val_list)

make_json(point_data, "train")
make_json(val_list, "val")


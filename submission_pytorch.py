from apis import init_detector, inference_detector, show_result

import numpy as np
import pandas as pd
import cv2
from tqdm import trange


config_file = 'faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'epoch_1.pth'
img_path = '/mnt/sda1/openimage2019/dataset/commit/test/'


# 初始化模型
model = init_detector(config_file, checkpoint_file)


# 预处理
label_csv = pd.read_csv("challenge-2019-classes-description-500.csv")

label_temp = pd.DataFrame({'label': label_csv.columns[0],
                           'name': label_csv.columns[1]},
                          index=[0])
label_csv.columns = ['label', 'name']
label_csv = label_temp.append(label_csv, ignore_index=True)
# print(label_csv)
num_to_label = {}
for i in range(len(label_csv)):
    num_to_label[label_csv.loc[i]['label']] = i
inverse_dic = {}
for key, val in num_to_label.items():
    inverse_dic[val] = key


def get_predictionstring(img):
    # 测试一张图片
    height = cv2.imread(img).shape[0]
    width = cv2.imread(img).shape[1]


    result = inference_detector(model, img)


    bbox_result, segm_result = result, None
    bboxes = np.row_stack(bbox_result)


    # 将bbox坐标归一化
    bboxes[:, 0] = bboxes[:, 0] / width
    bboxes[:, 1] = bboxes[:, 1] / height
    bboxes[:, 2] = bboxes[:, 2] / width
    bboxes[:, 3] = bboxes[:, 3] / height


    # 将bbox置信度放在第一列
    bboxes = bboxes[:, [4, 0, 1, 2, 3]]


    # 获取标签编号
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]

    # print(labels)
    labels = np.concatenate(labels)
    # print(labels)

    temp = []
    for label in labels:
        temp.append(inverse_dic[label])


    # 将标签和bbox数组按列拼接
    almost = np.column_stack((temp, bboxes))


    # 将数组展开成一维
    fla_almost = almost.flatten()


    PredictionString = ' '.join(fla_almost)
    return PredictionString


images = pd.read_csv('sample_submission.csv')

for i in trange(99999):
    image_name = img_path + images['ImageId'][i] + '.jpg'
    images['PredictionString'][i] = get_predictionstring(image_name)

images.to_csv("submission_test.csv", columns=('ImageId', 'PredictionString'), index=None)

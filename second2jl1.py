# import cv2
# import os
# import numpy as np
# import glob
# #1 耕地
# #2 道路
# #3 林草
# #4 建筑
# #5 其他
# pred1 = glob.glob(os.path.join(r"/data2/jw/semantic_cd/Bi-SRNet-main/PRED_CDSC_competation/im1", '*.png'))
# pred2 = glob.glob(os.path.join(r"/data2/jw/semantic_cd/Bi-SRNet-main/PRED_CDSC_competation/im2", '*.png'))

# for pred in pred1:
#     p1 = cv2.imread(pred,-1)
#     mask = np.zeros_like(p1)
#     name = os.path.basename(pred)
#     p2 = cv2.imread(os.path.join(r"/data2/jw/semantic_cd/Bi-SRNet-main/PRED_CDSC_competation/im2", name),-1)
#     mask = np.where((p1==1 and p2==2), 1, mask)
#     mask = np.where((p1==1 and p2==3), 2, mask)
#     mask = np.where((p1==1 and p2==4), 3, mask)
#     mask = np.where((p1==1 and p2==5), 4, mask)
#     mask = np.where((p1==2 and p2==1), 5, mask)
#     mask = np.where((p1==3 and p2==1), 6, mask)
#     mask = np.where((p1==4 and p2==1), 7, mask)
#     mask = np.where((p1==5 and p2==1), 8, mask)

import cv2
import os
import numpy as np
import glob

def read_images(folder_path):
    image_paths = glob.glob(os.path.join(folder_path, '*.png'))
    images = {os.path.basename(path): cv2.imread(path, -1) for path in image_paths}
    return images

def create_change_mask(image1, image2):
    mask = np.zeros_like(image1)
    conditions = [
        ((image1 == 1) & (image2 == 2), 1),
        ((image1 == 1) & (image2 == 3), 2),
        ((image1 == 1) & (image2 == 4), 3),
        ((image1 == 1) & (image2 == 5), 4),
        ((image1 == 2) & (image2 == 1), 5),
        ((image1 == 3) & (image2 == 1), 6),
        ((image1 == 4) & (image2 == 1), 7),
        ((image1 == 5) & (image2 == 1), 8),
        # ... 其他条件
    ]
    for condition, value in conditions:
        mask = np.where(condition, value, mask)
    return mask

# 读取图像
folder1 = "/data2/jw/semantic_cd/Bi-SRNet-main/PRED_CDSC_competation3/im1"
folder2 = "/data2/jw/semantic_cd/Bi-SRNet-main/PRED_CDSC_competation3/im2"
images1 = read_images(folder1)
images2 = read_images(folder2)

# 对每对图像生成变化掩码
for name, image1 in images1.items():
    image2 = images2.get(name)
    # print(image2)
    if image2 is not None:
        mask = create_change_mask(image1, image2)
        print(mask.shape)
        # 可以在这里保存或处理mask
        cv2.imwrite(os.path.join("/data2/jw/semantic_cd/Bi-SRNet-main/PRED_CDSC_competation3/results", name), mask)

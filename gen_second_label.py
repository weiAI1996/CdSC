import os
from skimage import io
import numpy as np
import glob
import cv2
import os
from skimage import io
import numpy as np
import glob
import cv2
tree=np.array([0,255,0])
building = np.array([128,0,0])
nvg = np.array([128,128,128])
water = np.array([0,0,255])
play_ground = np.array([255,0,0])
low_vegetation = np.array([0,128,0])

def color2changelabel(image1):
    rows, cols, _ = image1.shape
    labels = np.zeros((rows, cols), dtype=int)

    # 使用np.where进行灰度映射
    labels = np.where(np.all(image1 == water, axis=-1), 1, labels)
    labels = np.where(np.all(image1 == nvg, axis=-1), 2, labels)
    labels = np.where(np.all(image1 == low_vegetation, axis=-1), 3, labels)
    labels = np.where(np.all(image1 == tree, axis=-1), 4, labels)
    labels = np.where(np.all(image1 == building, axis=-1), 5, labels)
    labels = np.where(np.all(image1 == play_ground, axis=-1), 6, labels)
    return labels
if __name__ == '__main__':
    # imgs = glob.glob(os.path.join("/data/jw/semantic_cd/datasets/SECOND/test/label1", "*.png"))
    # for img in imgs:
    #     img1 = io.imread(img)
    #     img2path = img.replace("label1", "label2")
    #     img2 = io.imread(img2path)
    #     label = color2changelabel(img1)
    #     cv2.imwrite(img.replace("label1", "l1"), label)
    #     label = color2changelabel(img2)
    #     cv2.imwrite(img2path.replace("label2", "l2"), label)
    img = io.imread('/data2/jw/semantic_cd/datasets/SECOND/test/label1/00004.png')
    label = color2changelabel(img)
    cv2.imwrite("l.png",label)
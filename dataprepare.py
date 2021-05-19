import cv2
from utils import config
import ops
configg = config.Config()
import  numpy as np
def dataget(kk):
    img=cv2.imread(kk[0])
    sh=img.shape
    img1=cv2.resize(img,(448,448))
    positions=kk[2]
    newshape=ops.cul_reshape((sh[0],sh[1]),(448,448),np.array(positions,dtype=float))
    label = np.zeros((configg.featuersh, configg.featuersw, (configg.B*5+configg.classnum)))
    for i in range(0,kk[4]):
        p=newshape[i]
        y1=p[0]
        x1=p[1]
        y2 = p[2]
        x2 = p[3]
        boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
        x_ind = int(boxes[0] * configg.featuersw/ configg.image_size)
        y_ind = int(boxes[1] * configg.featuersh/configg.image_size)
        if label[y_ind, x_ind, 0] == 1:
            continue
        label[y_ind, x_ind, 0] = 1
        label[y_ind, x_ind, 1:5] = boxes
        label[y_ind, x_ind, 5 + kk[3][i][0]] = 1
    return img1,label
# if __name__ == '__main__':
# import vocget
#     path = './utils/VOC2007/VOCdevkit/VOC2007/Annotations'
#     path1 = './utils/VOC2007/VOCdevkit/VOC2007/JPEGImages'
#     xmlfile, count = vocget.getFileNames(configg.path)  # 得到该路径下有哪些xml文件
#     ms = vocget.getdataxy2(configg.path, configg.path1, 0, count - 1, xmlfile)
#     img,label=dataget(ms[2])
#     img=t

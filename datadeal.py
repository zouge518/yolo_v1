from utils.new_anchorcreate import *
import  cv2
from utils import config
import vocget
import utils.anchors as anchors
import  ops
configg=config.Config()
import  tensorflow as tf
if __name__ == '__main__':
    xmlfile, count = vocget.getFileNames(configg.path)  # 得到该路径下有哪些xml文件
    ms =vocget.getdataxy2(configg.path,configg.path1, 0, 5000, xmlfile)
    anchores = anchors.get_anchors([configg.featuersh, configg.featuersw], 600, 900)
    for i in range(0, len(ms)):
            # if(i!=99):
            kk = ms[i]
            img = cv2.imread(kk[0])
            names=str(kk[0]).split('/')[6].split('.')[0]
            print(names)
            shape1 = img.shape   #h,w
            img = cv2.resize(img, (900, 600))
            # 得到建议框，以及背景和非背景
            newshape =ops.cul_reshape((shape1[0], shape1[1]), (600, 900), np.array(kk[2]))
            anchor, lable= getbbx_train(anchores, newshape, threshold_max=0.6, threshold_min=0.3)
            w=tf.where(lable==1)
            anchor=np.array(anchor,float)
            lable = np.array(lable, 'int')
            np.save('./data1/anchor/'+str(names)+'.npy',anchor)
            np.save('./data1/label/' + str(names) + '.npy', lable)
            anchor1 = np.load('./data1/anchor/' + str(names) + '.npy', allow_pickle=True)
            lable1 = np.load('./data1/label/' + str(names) + '.npy', allow_pickle=True)
            print([anchor[w.numpy()]-anchor1[w.numpy()]])

            print()

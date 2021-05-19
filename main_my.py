import tensorflow as tf
from model import getrpnmodel
import numpy as np
import  cv2
from utils import config
import vocget
import  ops
from dataprepare import dataget
headnet,allmodel=getrpnmodel()
allmodel.summary()

configg=config.Config()

loadmode=True
if(loadmode):
    allmodel.load_weights(configg.modelpath)
    test=ops.test(allmodel)
else:
    test = ops.test(allmodel,loadmode=False)
loss=ops.cul_loss()
def yolotest(epoch):
    img = cv2.imread('./test/000009.jpg')
    img = cv2.resize(img, (448, 448))
    rr = img.copy()
    img1 = []
    img1.append(img)
    img = tf.convert_to_tensor(img1, dtype=tf.float32) / 125 - 1
    img=test.test1(img,rr)
    cv2.imwrite('./result/'+str(epoch)+'.jpg',img)
    return 0
def getdata(ms,numround=None):
    lables=[]
    imgs = []
    for i in numround:
        img, label = dataget(ms[i])
        imgs.append(img)
        lables.append(label)
    return imgs,lables
if __name__ == '__main__':
    Op = tf.keras.optimizers.Adam(learning_rate=1e-5)
    xmlfile, count = vocget.getFileNames(configg.path)  # 得到该路径下有哪些xml文件
    ms =vocget.getdataxy2(configg.path,configg.path1, 0, count - 1, xmlfile)
    number = np.random.randint(0, len(ms), len(ms))
    bathsize = configg.batchsize
    db = tf.data.Dataset.from_tensor_slices(number)  # 将数据特征与标记组合
    db = db.shuffle(4800).batch(batch_size=bathsize)
    for k in range(0,5000):
        count1 = 0
        for j in db:

                img,lable = getdata(ms, j)
                img=tf.convert_to_tensor(img,dtype=tf.float32)/125-1
                lable = tf.convert_to_tensor(lable, dtype=tf.float32)
                with tf.GradientTape() as typep:
                    msg=allmodel(img)
                    class_loss,object_loss , noobject_loss,coord_loss=loss.postion_loss(msg,lable)
                    loss1=class_loss+object_loss+noobject_loss+coord_loss
                    grad = typep.gradient(loss1, allmodel.trainable_variables)
                    Op.apply_gradients(zip(grad, allmodel.trainable_variables))
                    print(['epoch',count1,'allloss:',loss1.numpy(),'classloss:',class_loss.numpy(),'objecloss:',object_loss.numpy(),'noob:',noobject_loss.numpy(),'coloss:',coord_loss.numpy()])
                    if(count1%10==0):
                       headnet.save_weights('./model/head.ckpt')
                       allmodel.save_weights(configg.modelpath)
                       yolotest(count1)
                    count1 += 1

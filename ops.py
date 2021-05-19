import cv2
import tensorflow  as tf
from tensorflow.keras import layers
from  tensorflow import  keras
import numpy as np
from utils import  config
def conv2d_bn(x=None,filters=64, num_row=3,num_col=3,padding='same', strides=(2, 2),name=None,bn=True,activation='relu'):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = layers.Conv2D(filters=filters,kernel_size=(num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    if(bn):
        x = layers.BatchNormalization(axis=3, scale=False, name=bn_name)(x)
    if(activation=='relu'):
         x = layers.Activation(keras.activations.relu, name=name)(x)
    elif(activation=='linear'):
        x=layers.Activation(keras.activations.linear, name=name)(x)
    elif (activation == 'softmax'):
        x = layers.Activation(keras.activations.softmax, name=name)(x)
    return x

# 计算变形后的物体坐标
def cul_reshape(ori_rec=None,rel_rec=(600,900),shape=None):
    rh=np.array(rel_rec[0],dtype=float)
    rw = np.array(rel_rec[1],dtype=float)
    oh = np.array(ori_rec[0],dtype=float)
    ow = np.array(ori_rec[1],dtype=float)
    lw=rw/ow
    lh=rh/oh
    shape[:,0]=shape[:,0]*lh
    shape[:, 1] = shape[:, 1] * lw
    shape[:, 2] = shape[:, 2] *lh
    shape[:, 3] = shape[:, 3] *lw
    return shape

def calc_iou(boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

        boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
        rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[..., 2] * boxes1[..., 3]
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

class cul_loss():
    def __init__(self):
        configg=config.Config()
        self.img_size=configg.image_size
        self.cell_size = configg.featuersh
        self.boxes_per_cell = configg.B
        self.boundary1 = configg.classnum
        self.boundary2 = self.boundary1 +  self.boxes_per_cell*5
        self.class_scale = 1
        self.object_scale = 5
        self.noobject_scale = 0.5
        self.coord_scale = 1
        # self.offset = np.transpose(np.reshape(np.array(
        # [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
        # (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
    def postion_loss(self,predicts, labels):

        predict_classes = tf.reshape(predicts[..., self.boxes_per_cell*5:self.boundary2], [labels.shape[0], self.cell_size, self.cell_size, self.boundary1])
        predict_scales = tf.reshape(predicts[..., 0:self.boxes_per_cell], [labels.shape[0], self.cell_size, self.cell_size,  self.boxes_per_cell])
        predict_boxes = tf.reshape(predicts[..., self.boxes_per_cell:self.boxes_per_cell*5], [labels.shape[0], self.cell_size, self.cell_size, self.boxes_per_cell, 4])
        response = tf.reshape(labels[..., 0], [labels.shape[0], self.cell_size, self.cell_size, 1])
        boxes = tf.reshape(labels[..., 1:5], [labels.shape[0], self.cell_size, self.cell_size, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, 1, 1]) /self.img_size
        classes = labels[..., 5:]
        offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        offset = tf.reshape(tf.constant( offset, dtype=tf.float32), [1, self.cell_size, self.cell_size, 1])
        offset = tf.tile(offset, [labels.shape[0], 1, 1, 1])
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))
        # print([offset,offset_tran])
        predict_boxes_tran = tf.stack(
            [(predict_boxes[..., 0] + offset) /  self.cell_size,
             (predict_boxes[..., 1] + offset_tran) /  self.cell_size,
             tf.square(predict_boxes[..., 2]),
             tf.square(predict_boxes[..., 3])], axis=-1)

        iou_predict_truth = calc_iou(predict_boxes_tran, boxes)

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keepdims=True)
        object_mask = tf.cast(
            (iou_predict_truth >= object_mask), tf.float32) * response

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(
            object_mask, dtype=tf.float32) - object_mask
        boxes_tran = tf.stack(
            [boxes[..., 0] *  self.cell_size - offset,
             boxes[..., 1] *  self.cell_size - offset_tran,
             tf.sqrt(boxes[..., 2]),
             tf.sqrt(boxes[..., 3])], axis=-1)

        # class_loss
        class_delta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
            name='class_loss') *  self.class_scale

        # object_loss
        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
            name='object_loss') *  self.object_scale

        # noobject_loss
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
            name='noobject_loss') *  self.noobject_scale

        # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
            name='coord_loss') *  self.coord_scale
        # loss = class_loss + object_loss + noobject_loss + coord_loss
        return class_loss,object_loss , noobject_loss , coord_loss

class test():
    def __init__(self,model,loadmode=True):
        self.allmodel = model
        self.configg = config.Config()
        self.cell_size = self.configg.featuersh
        self.boxes_per_cell = self.configg.B
        self.boundary1 = self.configg.classnum
        self.boundary2 = self.boundary1 + self.boxes_per_cell * 5
        self.threshold=0.1
        self.iou_threshold=0.6
        self.classes = self.configg.CLASSES
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        if (loadmode == True):
            self.allmodel.load_weights(self.configg.modelpath)
    def test1(self,img,img1):
        output=self.allmodel(img)
        result=self.interpret_output(output)
        # for i in range(len(result)):
        #     result[i][1] *= (1.0 *  self.configg.image_size)
        #     result[i][2] *= (1.0 *  self.configg.image_size)
        #     result[i][3] *= (1.0 *  self.configg.image_size)
        #     result[i][4] *= (1.0 *  self.configg.image_size)
        for i in range(len(result)):
            x = int(result[i][0])
            y = int(result[i][1])
            w = int(result[i][2] / 2)
            h = int(result[i][3] / 2)
            COLORS = np.random.randint(0, 255, (1, 3))
            cv2.rectangle(img1, (x - w, y - h), (x + w, y + h), (np.int(COLORS[0][0]),np.int(COLORS[0][1]),np.int(COLORS[0][2])), 2)
            # cv2.rectangle(img1, (x - w, y - h - 20),
            #               (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img1,result[i][5] + ' : %.2f' % result[i][4],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, lineType)
        return img1

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.boundary1))

        class_probs = output[...,self.boxes_per_cell*5:self.boundary2][0].numpy()
        scales = np.reshape(
            output[...,0:self.boxes_per_cell],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(
            output[...,self.boxes_per_cell:self.boxes_per_cell*5],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))

        offsets = np.transpose(
            np.reshape(self.offset,[self.boxes_per_cell, self.cell_size, self.cell_size]),(1, 2, 0)
                )
        boxes1 = boxes.copy()
        boxes1[...,0] = boxes1[...,0] +offsets
        boxes1[:, :, :, 1] = boxes1[:, :, :, 1] +np.transpose(offsets, (1, 0, 2))
        boxes1[:, :, :, :2] = 1.0 * boxes1[:, :, :, 0:2] / self.cell_size
        boxes1[:, :, :, 2:] = np.square(boxes1[:, :, :, 2:])

        boxes1 =boxes1*self.configg.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.boundary1):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes1[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if calc_iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]
        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i],
                    self.classes[classes_num_filtered[i]]]
                 )
        return result
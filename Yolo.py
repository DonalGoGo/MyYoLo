

import tensorflow as tf;
import numpy as np;
from config import *
slim = tf.contrib.slim


class Yolo(object):
    def __init__(self):
        self.batch_size=BATCH_SIZE
        self.cell_size=CELL_SIZE
        self.boxes_per_cell=BOXES_PER_CELL
        self.image_size=IMAGE_SIZE
        self.pred_dim_per_box=PRED_DIM_PER_BOX
        self.class_num=CLASS_NUM
        self.class_lambda = 2.0
        self.object_lambda=2.0
        self.noobject_lambda=2.0
        self.coord_lambda=5.0

        #self.split_boundary1主要用来分离最后的预测类别结果，提取出预测的class
        self.split_boundary1=self.cell_size*self.cell_size*self.class_num
        # self.split_boundary1主要用来分离最后的预测自信度和bounding box
        self.split_boundary2=self.cell_size*self.cell_size*self.boxes_per_cell+self.split_boundary1

        temp0=np.array([np.arange(self.cell_size)]*self.cell_size*self.boxes_per_cell)
        temp1=np.reshape(temp0,[self.cell_size,self.boxes_per_cell,self.cell_size])
        self.offset=np.transpose(temp1,axes=[0,2,1])

        self.input_x = tf.placeholder(tf.uint8, shape=[None, self.image_size, self.image_size, 1], name="input_x")
        self.real_y = tf.placeholder(tf.float32, shape=[None, 7, 7, 25], )
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.yolo_net(self.input_x, self.is_train,scope='yolo')
        self.calLoss(self.real_y,self.pred_y)
        self.total_loss=tf.losses.get_total_loss()
        loss=tf.losses.get_losses()
        re_loss=tf.losses.get_regularization_losses()
        tf.summary.scalar("total_loss",self.total_loss)

    def leaky_relu(self,alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')

        return op
    def yolo_net(self,input_x, is_train,scope='yolo'):
        images = input_x / 225 - 0.5;
        with tf.variable_scope(scope):
            batch_norm_params={
              'decay': 0.95,
              'is_training': is_train
            }
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=self.leaky_relu(0.1),
                                #normalizer_fn=slim.batch_norm,
                                #normalizer_params=batch_norm_params,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                input = slim.conv2d(images, 64, 7, 2, padding="SAME", scope="conv_1")
                input = slim.max_pool2d(input, 2, padding="SAME", scope="pool_2")

                input = slim.conv2d(input, 192, 3, 1, padding="SAME", scope="conv_3")
                input = slim.max_pool2d(input, 2, padding="SAME", scope="pool_4")

                input = slim.conv2d(input, 128, 1, 1, padding="SAME", scope="conv_5")
                input = slim.conv2d(input, 256, 3, 1, padding="SAME", scope="conv_6")
                input = slim.conv2d(input, 256, 1, 1, padding="SAME", scope="conv_7")
                input = slim.conv2d(input, 512, 3, 1, padding="SAME", scope="conv_8")
                input = slim.max_pool2d(input, 2, padding="SAME", scope="pool_9")

                for i in range(4):
                    input = slim.conv2d(input, 256, 1, 1, padding="SAME", scope="conv_"+str(10+2*i))
                    input = slim.conv2d(input, 512, 3, 1, padding="SAME", scope="conv_"+str(10+2*i+1))
                input = slim.conv2d(input, 512, 1, 1, padding="SAME", scope="conv_18")
                input = slim.conv2d(input, 1024, 3, 1, padding="SAME", scope="conv_19")
                input = slim.max_pool2d(input, 2, padding="SAME", scope="pool_20")

                for i in range(2):
                    input = slim.conv2d(input, 512, 1, 1, padding="SAME", scope="conv_"+str(20+2*i+1))
                    input = slim.conv2d(input, 1024, 3, 1, padding="SAME", scope="conv_"+str(20+2*i+2))
                input = slim.conv2d(input, 1024, 3, 1, padding="SAME", scope="conv_25")
                input = slim.conv2d(input, 1024, 3, 2, padding="SAME", scope="conv_26")

                input = slim.conv2d(input, 1024, 3, 1, padding="SAME", scope="conv_27")
                input = slim.conv2d(input, 1024, 3, 1, padding="SAME", scope="conv_28")

                input=tf.transpose(input,[0,3,1,2],name="trans_29")
                input=slim.flatten(input,scope="flat_30")
                input=slim.fully_connected(input,4096,scope="fc_31")
                output=slim.fully_connected(input,1470)
                self.pred_y=output;
                return output


    def myYoloNet(self,input_x, is_train):
        tempinput = input_x / 225 - 0.5;
        with tf.variable_scope("layer1"):
            tempinput = tf.layers.conv2d(tempinput, filters=64, kernel_size=7, strides=2,padding="SAME");
            tempinput = tf.layers.batch_normalization(tempinput, training=is_train);
            tempinput = tf.nn.leaky_relu(tempinput, alpha=0.1);
            tempinput = tf.layers.max_pooling2d(tempinput, 2, 2);
        with tf.variable_scope("layer2"):
            tempinput = tf.layers.conv2d(tempinput, filters=192, kernel_size=3,padding="SAME");
            tempinput = tf.layers.batch_normalization(tempinput, training=is_train);
            tempinput = tf.nn.leaky_relu(tempinput, alpha=0.1);
            tempinput = tf.layers.max_pooling2d(tempinput, 2, 2);
        with tf.variable_scope("layer3"):
            tempinput = self.convGroup1_3(tempinput, 128, is_train);
            tempinput = self.convGroup1_3(tempinput, 256, is_train);
            tempinput = tf.layers.max_pooling2d(tempinput, 2, 2);
        with tf.variable_scope("layer4"):
            for i in range(4):
                tempinput = self.convGroup1_3(tempinput, 256, is_train);
            tempinput = self.convGroup1_3(tempinput, 512, is_train);
            tempinput = tf.layers.max_pooling2d(tempinput, 2, 2);
        with tf.variable_scope("layer5"):
            for i in range(2):
                tempinput = self.convGroup1_3(tempinput, 512, is_train);
            tempinput = tf.layers.conv2d(tempinput, filters=1024, kernel_size=3,padding="SAME");
            tempinput = tf.layers.batch_normalization(tempinput, training=is_train);
            tempinput = tf.nn.leaky_relu(tempinput, alpha=0.1);
            tempinput = tf.layers.conv2d(tempinput, filters=1024, kernel_size=3, strides=2,padding="SAME");
            tempinput = tf.layers.batch_normalization(tempinput, training=is_train);
            tempinput = tf.nn.leaky_relu(tempinput, alpha=0.1);
        with tf.variable_scope("layer6"):
            for i in range(2):
                tempinput = tf.layers.conv2d(tempinput, filters=1024, kernel_size=3,padding="SAME");
                tempinput = tf.layers.batch_normalization(tempinput, training=is_train);
                tempinput = tf.nn.leaky_relu(tempinput, alpha=0.1);
        print("final conv shape=", tempinput.shape);
        batchs, height, width, channels = tempinput.shape;
        fc1_size = int(height * width * channels);
        tempinput = tf.reshape(tempinput, [-1, fc1_size], name="fc1");
        tf.layers.dense(tempinput, units=4096);
        tempinput = tf.layers.batch_normalization(tempinput, training=is_train);
        tempinput = tf.nn.leaky_relu(tempinput, alpha=0.1);
        result = tf.layers.dense(tempinput, units=7 * 7 * 30);
        #result = tf.reshape(result, shape=[None, 7 * 7, 30])
        return result;



    def convGroup1_3(self,tempinput,filter_num,is_train):
        tempinput = tf.layers.conv2d(tempinput, filters=filter_num, kernel_size=1,padding="SAME");
        tempinput = tf.layers.batch_normalization(tempinput, training=is_train);
        tempinput = tf.nn.leaky_relu(tempinput, alpha=0.1);
        tempinput = tf.layers.conv2d(tempinput, filters=filter_num*2, kernel_size=3,padding="SAME");
        tempinput = tf.layers.batch_normalization(tempinput, training=is_train);
        tempinput = tf.nn.leaky_relu(tempinput, alpha=0.1);
        return tempinput;




    def calLoss(self,real_y,pred_y,scope='loss_layer'):
        with tf.variable_scope(scope):
            #xy坐标损失
            respone=tf.reshape(real_y[:,:,:,0],shape=[self.batch_size,self.cell_size,self.cell_size,1])
            real_box=tf.reshape(real_y[:,:,:,1:5],shape=[self.batch_size,self.cell_size,self.cell_size,1,4])
            real_box=tf.tile(real_box,[1,1,1,self.boxes_per_cell,1])/self.image_size
            real_class=real_y[:,:,:,5:]

            pred_class=tf.reshape(pred_y[:,:self.split_boundary1],shape=[self.batch_size,self.cell_size,self.cell_size,self.class_num])
            pred_conf=tf.reshape(pred_y[:,self.split_boundary1:self.split_boundary2],shape=[self.batch_size,self.cell_size,self.cell_size,self.boxes_per_cell])
            pred_box=tf.reshape(pred_y[:,self.split_boundary2:],shape=[self.batch_size,self.cell_size,self.cell_size,self.boxes_per_cell,4])

            offset=tf.constant(self.offset,dtype=tf.float32)
            offset=tf.reshape(offset,shape=[1,self.cell_size,self.cell_size,self.boxes_per_cell])
            offset=tf.tile(offset,[self.batch_size,1,1,1])

            pred_box0=pred_box[:,:,:,:,0]+offset
            pred_box1=pred_box[:,:,:,:,1]
            trans_offset=tf.transpose(offset,(0,2,1,3))
            pred_box1=pred_box1+trans_offset

            pred_box_trans=tf.stack([pred_box0/self.cell_size,
                                     pred_box1/self.cell_size,
                                     tf.square(pred_box[:,:,:,:,2]),
                                     tf.square(pred_box[:,:,:,:,3])])
            pred_box_trans=tf.transpose(pred_box_trans,(1,2,3,4,0))

            iou_pred_truth=self.calc_iou(pred_box_trans,real_box)

            object_max=tf.reduce_max(iou_pred_truth,3,keep_dims=True)
            object_mask=tf.cast((iou_pred_truth>=object_max),tf.float32)*respone

            noobject_mask=tf.ones_like(object_mask,dtype=tf.float32)-object_mask

            real_box_trans=tf.stack([real_box[:,:,:,:,0]*self.cell_size-offset,
                                     real_box[:,:,:,:,1]*self.cell_size-trans_offset,
                                     tf.sqrt(real_box[:,:,:,:,2]),
                                     tf.sqrt(real_box[:,:,:,:,3])])
            real_box_trans=tf.transpose(real_box_trans,(1,2,3,4,0))

            object_loss=object_mask*(pred_conf-iou_pred_truth)
            object_loss=tf.reduce_mean(tf.reduce_sum(tf.square(object_loss),axis=[1,2,3]),name="object_loss")*self.object_lambda

            noobject_loss=noobject_mask*pred_conf
            noobject_loss=tf.reduce_mean(tf.reduce_sum(tf.square(noobject_loss),axis=[1,2,3]),name="noobject_loss")*self.noobject_lambda

            coord_mask=tf.expand_dims(object_mask,4)
            coord_loss=coord_mask*(pred_box-real_box_trans)
            coord_loss=tf.reduce_mean(tf.reduce_sum(tf.square(coord_loss),axis=[1,2,3,4]),name="coord_loss")*self.coord_lambda

            class_loss_temp0=respone*(pred_class-real_class)
            per_image_class_loss=tf.reduce_sum(tf.square(class_loss_temp0),axis=[1,2,3])
            class_loss=tf.reduce_mean(per_image_class_loss,name="class_loss")*self.class_lambda

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(coord_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)

            tf.summary.scalar("class_loss",class_loss)
            tf.summary.scalar("coord_loss", coord_loss)
            tf.summary.scalar("object_loss", object_loss)
            tf.summary.scalar("noobject_loss", noobject_loss)

            tf.summary.histogram('iou', iou_pred_truth)









    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def main():
    yolo=Yolo()
    pred_y=tf.Variable(tf.zeros([20,1470]))
    real_y=tf.Variable(tf.zeros([20,7,7,25]))
    sum_loss=yolo.calLoss(real_y,pred_y)







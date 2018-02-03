import tensorflow as tf
import numpy as np
from config import *
import datetime
import os
from ReadData import *

slim=tf.contrib.slim

class Solver(object):
    def __init__(self,yolo):
        self.yolo=yolo
        self.epoch=EPOCH
        self.min_loss=1000;
        self.init_learning_rate=0.05
        self.decay_rate=0.95
        self.decay_step=500
        self.staircase=True
        self.variable_restore=tf.global_variables();
        self.restore=tf.train.Saver(self.variable_restore,max_to_keep=5)
        self.saver=tf.train.Saver(self.variable_restore,max_to_keep=5)

        self.output_dir=os.path.join("data/pascal_voc/output/", datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
        self.summary_op=tf.summary.merge_all()
        self.writer=tf.summary.FileWriter(self.output_dir,flush_secs=60)
        self.global_step=tf.Variable(tf.constant(0),trainable=False,name="global_step")
        self.learning_rate=tf.train.exponential_decay(self.init_learning_rate,self.global_step,self.decay_step,self.decay_rate,self.staircase,name="learning_rate")
        self.optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.yolo.total_loss,global_step=self.global_step)
        self.ema=tf.train.ExponentialMovingAverage(decay=0.9999)
        self.averages_op=self.ema.apply(tf.trainable_variables())
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([self.optimizer]):
            self.train_op=tf.group(self.averages_op)
        gpu_options=tf.GPUOptions()
        gpu_config=tf.ConfigProto(gpu_options=gpu_options)
        self.sess=tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.writer.add_graph(self.sess.graph)

    def train(self):
        ImgPath = 'data/VOC2007/JPEGImages/';
        AnnoPath = 'data/VOC2007/Annotations/';
        train_set, valid_set = splitTrainAndValid(AnnoPath)
        count=1;
        for i in range(EPOCH):
            for x_batch, y_batch in generateFinalDataSet(ImgPath, AnnoPath, train_set, BATCH_SIZE):
                if(count%10==0):
                    pred_loss,current_learn_rate,global_step,summary_str=self.sess.run([self.yolo.total_loss,self.learning_rate,self.global_step,self.summary_op],feed_dict={self.yolo.input_x:x_batch,self.yolo.real_y:y_batch,self.yolo.is_train:False})
                    print("step=%d pred_loss=%f ,learn_rate=%f,global_step=%f"%(count,pred_loss,current_learn_rate,global_step))
                    if(pred_loss<self.min_loss and count>500):
                        self.min_loss=pred_loss
                        self.saver.save(self.sess,self.ckpt_file,global_step=self.global_step)
                    self.writer.add_summary(summary_str)
                else:
                    self.sess.run([self.train_op, self.extra_update_ops],
                                  feed_dict={self.yolo.input_x: x_batch, self.yolo.real_y: y_batch,
                                             self.yolo.is_train: True})
                count += 1







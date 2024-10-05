
from layer import *
from data_loader import dataloader
from vgg19 import Vgg19
import face_68points
import feature_compute

#unet 部分，偷懒

import os
import tensorflow as tf
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
import scipy.io
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation

 
 
class DeblurGAN():
    
    def __init__(self, args):
        
        self.data_loader = dataloader(args)
        print("data has been loaded")
 
        self.channel = 3
 
        self.n_feats = args.n_feats
        self.mode = args.mode
        self.batch_size = args.batch_size      
        self.num_of_down_scale = args.num_of_down_scale
        self.gen_resblocks = args.gen_resblocks
        self.discrim_blocks = args.discrim_blocks
        self.vgg_path = args.vgg_path
        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step
        self.fine_tuning = args.fine_tuning
        self.encode = args.encode

        
    def down_scaling_feature(self, name, x, n_feats):
        x = Conv(name=name + 'conv', x=x, filter_size=3, in_filters=n_feats,
                 out_filters=n_feats * 2, strides=2, padding='SAME')
        x = instance_norm(x)
        x = tf.nn.relu(x)
        
        return x
    
    def up_scaling_feature(self, name, x, n_feats):
        x = Conv_transpose(name=name + 'deconv', x=x, filter_size=3, in_filters=n_feats,
                           out_filters=n_feats // 2, fraction=2, padding='SAME')
        x = instance_norm(x)
        x = tf.nn.relu(x)
        
        return x
    
    def res_block(self, name, x, n_feats):
        
        _res = x
        
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = Conv(name=name + 'conv1', x=x, filter_size=3, in_filters=n_feats,
                 out_filters=n_feats, strides=1, padding='VALID')
        x = instance_norm(x)
        x = tf.nn.relu(x)
        
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = Conv(name=name + 'conv2', x=x, filter_size=3, in_filters=n_feats,
                 out_filters=n_feats, strides=1, padding='VALID')
        x = instance_norm(x)
        
        x = x + _res
        
        return x
    
    def generator(self, x, reuse=False, name='generator'):
        
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            _res = x

            x2 = x

            #x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')

            c = 0
            while(c<2):
                self.uconv1_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
                self.uBatchNorm1_1 = instance_norm(self.uconv1_1)
                """self.uBatchNorm1_1 = BatchNormalization(axis=3, gamma_regularizer=l2(self.learning_rate), beta_regularizer=l2(self.learning_rate))(
                    self.uconv1_1)"""
                self.uReLU1_1 = Activation('relu')(self.uBatchNorm1_1)
                self.uconv1_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uReLU1_1)
                self.udrop1_2 = Dropout(0)(self.uconv1_2)
                self.uMerge1 = concatenate([self.uconv1_1, self.udrop1_2], axis=3)
                self.upool1 = MaxPooling2D(pool_size=(2, 2))(self.uMerge1)

                self.uconv2_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.upool1)
                self.uBatchNorm2_1 = instance_norm(self.uconv2_1)
                """self.uBatchNorm2_1 = BatchNormalization(axis=3, gamma_regularizer=l2(self.learning_rate), beta_regularizer=l2(self.learning_rate))(
                    self.uconv2_1)"""
                self.uReLU2_1 = Activation('relu')(self.uBatchNorm2_1)
                self.uconv2_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uReLU2_1)
                self.udrop2_2 = Dropout(0)(self.uconv2_2)
                self.uMerge2 = concatenate([self.uconv2_1, self.udrop2_2], axis=3)
                self.upool2 = MaxPooling2D(pool_size=(2, 2))(self.uMerge2)

                self.uconv3_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.upool2)
                self.uBatchNorm3_1 = instance_norm(self.uconv3_1)
                """self.uBatchNorm3_1 = BatchNormalization(axis=3, gamma_regularizer=l2(self.learning_rate), beta_regularizer=l2(self.learning_rate))(
                    self.uconv3_1)"""
                self.uReLU3_1 = Activation('relu')(self.uBatchNorm3_1)
                self.uconv3_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uReLU3_1)
                self.udrop3_2 = Dropout(0)(self.uconv3_2)
                self.uMerge3 = concatenate([self.uconv3_1, self.udrop3_2], axis=3)
                self.upool3 = MaxPooling2D(pool_size=(3, 3))(self.uMerge3)  # 3,3

                self.uconv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.upool3)
                self.uBatchNorm4_1 = instance_norm(self.uconv4_1)
                """self.uBatchNorm4_1 = BatchNormalization(axis=3, gamma_regularizer=l2(self.learning_rate), beta_regularizer=l2(self.learning_rate))(
                    self.uconv4_1)"""
                self.uReLU4_1 = Activation('relu')(self.uBatchNorm4_1)
                self.uconv4_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uReLU4_1)
                self.udrop4_2 = Dropout(0)(self.uconv4_2)
                self.uMerge4 = concatenate([self.uconv4_1, self.udrop4_2], axis=3)
                self.udrop4 = Dropout(0.5)(self.uMerge4)
                self.upool4 = MaxPooling2D(pool_size=(5, 5))(self.udrop4)  # 5,5

                self.uconv5_1 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.upool4)
                self.uBatchNorm5_1 = instance_norm(self.uconv5_1)
                """self.uBatchNorm5_1 = BatchNormalization(axis=3, gamma_regularizer=l2(self.learning_rate), beta_regularizer=l2(self.learning_rate))(
                    self.uconv5_1)"""
                self.uReLU5_1 = Activation('relu')(self.uBatchNorm5_1)
                self.uconv5_2 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uReLU5_1)
                self.udrop5_2 = Dropout(0)(self.uconv5_2)
                self.uMerge5 = concatenate([self.uconv5_1, self.udrop5_2], axis=3)
                self.udrop5 = Dropout(0.5)(self.uMerge5)

                self.uup6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                    UpSampling2D(size=(5, 5))(self.udrop5))  # 5,5
                self.umerge6 = concatenate([self.udrop4, self.uup6], axis=3)
                self.uconv6_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.umerge6)
                self.uBatchNorm6_1 = instance_norm(self.uconv6_1)
                """self.uBatchNorm6_1 = BatchNormalization(axis=3, gamma_regularizer=l2(self.learning_rate), beta_regularizer=l2(self.learning_rate))(
                    self.uconv6_1)"""
                self.uReLU6_1 = Activation('relu')(self.uBatchNorm6_1)
                self.uconv6_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uReLU6_1)
                self.udrop6_2 = Dropout(0)(self.uconv6_2)
                self.uMerge6 = concatenate([self.uconv6_1, self.udrop6_2], axis=3)

                self.uup7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                    UpSampling2D(size=(3, 3))(self.uMerge6))  # 3,3
                self.umerge7 = concatenate([self.uMerge3, self.uup7], axis=3)
                self.uconv7_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.umerge7)
                self.uBatchNorm7_1 = instance_norm(self.uconv7_1)
                """self.uBatchNorm7_1 = BatchNormalization(axis=3, gamma_regularizer=l2(self.learning_rate), beta_regularizer=l2(self.learning_rate))(
                    self.uconv7_1)"""
                self.uReLU7_1 = Activation('relu')(self.uBatchNorm7_1)
                self.uconv7_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uReLU7_1)
                self.udrop7_2 = Dropout(0)(self.uconv7_2)
                self.uMerge7 = concatenate([self.uconv7_1, self.udrop7_2], axis=3)

                self.uup8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                    UpSampling2D(size=(2, 2))(self.uMerge7))
                self.umerge8 = concatenate([self.uMerge2, self.uup8], axis=3)
                self.uconv8_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.umerge8)
                self.uBatchNorm8_1 = instance_norm(self.uconv8_1)
                """self.uBatchNorm8_1 = BatchNormalization(axis=3, gamma_regularizer=l2(self.learning_rate), beta_regularizer=l2(self.learning_rate))(
                    self.uconv8_1)"""
                self.uReLU8_1 = Activation('relu')(self.uBatchNorm8_1)
                self.uconv8_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uReLU8_1)
                self.udrop8_2 = Dropout(0)(self.uconv8_2)
                self.uMerge8 = concatenate([self.uconv8_1, self.udrop8_2], axis=3)

                self.uup9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                    UpSampling2D(size=(2, 2))(self.uMerge8))
                self.umerge9 = concatenate([self.uMerge1, self.uup9], axis=3)
                self.uconv9_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.umerge9)
                self.uBatchNorm9_1 = instance_norm(self.uconv9_1)
                """self.uBatchNorm9_1 = BatchNormalization(axis=3, gamma_regularizer=l2(self.learning_rate), beta_regularizer=l2(self.learning_rate))(
                    self.uconv9_1)"""
                self.uReLU9_1 = Activation('relu')(self.uBatchNorm9_1)
                self.uconv9_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uReLU9_1)
                self.udrop9_2 = Dropout(0)(self.uconv9_2)
                self.uMerge9 = concatenate([self.uconv9_1, self.udrop9_2], axis=3)

                self.uconv9 = Conv2D(2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                    self.uMerge9)  # 3,3
                x = Conv2D(3, 1, activation='sigmoid')(self.uconv9)  # sigmoid
                c += 1







            x2 = tf.pad(x2, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')

            x2 = Conv(name='conv1', x=x2, filter_size=7, in_filters=self.channel,
                     out_filters=self.n_feats, strides=1, padding='VALID')
            # x = instance_norm(name = 'inst_norm1', x = x, dim = self.n_feats)
            x2 = instance_norm(x2)
            x2 = tf.nn.relu(x2)

            for i in range(self.num_of_down_scale):
                x2 = self.down_scaling_feature(name='down_%02d' % i, x=x2, n_feats=self.n_feats * (i + 1))

            for i in range(self.gen_resblocks):
                x2 = self.res_block(name='res_%02d' % i, x=x2, n_feats=self.n_feats * (2 ** self.num_of_down_scale))

            for i in range(self.num_of_down_scale):
                x2 = self.up_scaling_feature(name='up_%02d' % i, x=x2,
                                            n_feats=self.n_feats * (2 ** (self.num_of_down_scale - i)))

            x2 = tf.pad(x2, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
            x2 = Conv(name='conv_last', x=x2, filter_size=7, in_filters=self.n_feats,
                     out_filters=self.channel, strides=1, padding='VALID')
            x2 = tf.nn.tanh(x2)




            x =  0.5*_res + x + 0.5*x2
            x = tf.clip_by_value(x, -1.0, 1.0)

            return x

    def discriminator(self, x, reuse=False, name='discriminator'):

        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            x = Conv(name='conv1', x=x, filter_size=4, in_filters=self.channel,
                     out_filters=self.n_feats, strides=2, padding="SAME")
            x = instance_norm(x)
            x = tf.nn.leaky_relu(x)

            n = 1

            for i in range(self.discrim_blocks):
                prev = n
                n = min(2 ** (i+1), 8)
                x = Conv(name='conv%02d' % i, x=x, filter_size=4, in_filters=self.n_feats * prev,
                         out_filters=self.n_feats * n, strides=2, padding="SAME")
                x = instance_norm(x)
                x = tf.nn.leaky_relu(x)
                
            prev = n
            n = min(2 ** self.discrim_blocks, 8)
            x = Conv(name='conv_d1', x=x, filter_size=4, in_filters=self.n_feats * prev,
                     out_filters=self.n_feats * n, strides=1, padding="SAME")
            # x = instance_norm(name = 'instance_norm_d1', x = x, dim = self.n_feats * n)
            x = instance_norm(x)
            x = tf.nn.leaky_relu(x)
            
            x = Conv(name='conv_d2', x=x, filter_size=4, in_filters=self.n_feats * n,
                     out_filters=1, strides=1, padding="SAME")
            x = tf.nn.sigmoid(x)
            
            return x
    
        
    def build_graph(self):
        
        # if self.in_memory:
        self.blur = tf.placeholder(name="blur", shape=[None, None, None, self.channel], dtype=tf.float32)
        self.sharp = tf.placeholder(name="sharp", shape=[None, None, None, self.channel], dtype=tf.float32)

        x = self.blur
        label = self.sharp
        
        self.epoch = tf.placeholder(name='train_step', shape=None, dtype=tf.int32)
        self.deference = tf.placeholder(name='deference', shape=None, dtype=tf.float32)
        self.distance = tf.placeholder(name='distance', shape=None, dtype=tf.float32)
        
        x = (2.0 * x / 255.0) - 1.0
        label = (2.0 * label / 255.0) - 1.0

        
        self.gene_img = self.generator(x, reuse=False)
        self.real_prob = self.discriminator(label, reuse=False)
        self.fake_prob = self.discriminator(self.gene_img, reuse=True)
        
        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        
        interpolated_input = epsilon * label + (1 - epsilon) * self.gene_img
        gradient = tf.gradients(self.discriminator(interpolated_input, reuse=True), [interpolated_input])[0]
        GP_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gradient), axis=[1, 2, 3])) - 1))
        
        d_loss_real = - tf.reduce_mean(self.real_prob)
        d_loss_fake = tf.reduce_mean(self.fake_prob)

        self.vgg_net = Vgg19(self.vgg_path)
        self.vgg_net.build(tf.concat([label, self.gene_img], axis=0))
        #self.content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(
            #self.vgg_net.relu3_3[self.batch_size:] - self.vgg_net.relu3_3[:self.batch_size]), axis=3))
        self.content_loss = tf.reduce_mean(tf.reduce_mean(tf.square(
            self.vgg_net.relu3_3[self.batch_size:] - self.vgg_net.relu3_3[:self.batch_size]), axis=3))
        self.temp_loss = tf.reduce_mean(tf.reduce_mean(tf.square(
            self.vgg_net.relu5_4[self.batch_size:] - self.vgg_net.relu5_4[:self.batch_size]), axis=3))
        self.unet_loss = tf.reduce_mean(tf.reduce_mean(tf.square(
            self.vgg_net.uconv10[self.batch_size:] - self.vgg_net.uconv10[:self.batch_size]), axis=3))
        self.sobel_loss = tf.reduce_mean(tf.reduce_mean(tf.square(
            self.vgg_net.grad_components[self.batch_size:] - self.vgg_net.grad_components[:self.batch_size]), axis=-1))


        
        self.D_loss = d_loss_real + d_loss_fake + 10.0 * GP_loss
        #self.G_loss = - d_loss_fake + 100.0 * self.content_loss + 100.0 * tf.cast(self.deference, tf.float32) + 1000.0*tf.cast(self.distance, tf.float32)
        self.G_loss = - d_loss_fake + 100.0 * self.content_loss  +10.0*self.temp_loss + 1000.0*self.unet_loss + 100.0 * tf.cast(self.deference, tf.float32) + 1000.0*tf.cast(self.distance, tf.float32) + 0.5*self.sobel_loss


        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        lr = tf.minimum(self.learning_rate, tf.abs(2 * self.learning_rate - (
                self.learning_rate * tf.cast(self.epoch, tf.float32) / self.decay_step)))
        """
        G_var_to_tuning = [val for val in t_vars if self.encode in val.name and 'generator' in val.name]


        if self.fine_tuning:
            self.D_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.D_loss, var_list=D_vars)
            self.G_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.G_loss, var_list=G_var_to_tuning)
        else:
        """
        self.D_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.D_loss, var_list=D_vars)
        self.G_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.G_loss, var_list=G_vars)

        
        self.PSNR = tf.reduce_mean(tf.image.psnr(((self.gene_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val=1.0))
        self.ssim = tf.reduce_mean(tf.image.ssim(((self.gene_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val=1.0))
        
        logging_D_loss = tf.summary.scalar(name='D_loss', tensor=self.D_loss)
        logging_G_loss = tf.summary.scalar(name='G_loss', tensor=self.G_loss)
        logging_PSNR = tf.summary.scalar(name='PSNR', tensor=self.PSNR)
        logging_ssim = tf.summary.scalar(name='ssim', tensor=self.ssim)



 
        self.output = (self.gene_img + 1.0) * 255.0 / 2.0
        self.output = tf.round(self.output)
        self.output = tf.cast(self.output, tf.uint8)

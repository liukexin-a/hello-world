import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import face_68points as fp
import feature_compute as fc
import cv2
import extract_feature as ef

import util
 
 
def train(args, model, sess, saver):
    if args.fine_tuning:
        checkpoint_file = tf.train.latest_checkpoint(args.pre_trained_model)
        print("model path is %s" % checkpoint_file)
        saver.restore(sess, checkpoint_file)
        print("saved model is loaded for fine-tuning!")

    num_imgs = len(os.listdir(args.train_Sharp_path))

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs', sess.graph)



    """
    if args.fine_tuning:

        #获得所有参数global_variables();获得可以训练的参数trainable_variables()
        var = tf.trainable_variables()

        #选择要进行更新的参数
        var_to_train = [val for val in var if args.encode in val.name]

        #提取固定的不更新的参数
        var_to_restore = [val for val in var if val not in var_to_train]


        saver = tf.train.Saver(var_to_restore)
        checkpoint_file = tf.train.latest_checkpoint(args.pre_trained_model)
        print("model path is %s" % checkpoint_file)
        saver.restore(sess, checkpoint_file)
        print("saved model is loaded for fine-tuning!")


        tf.initialize_variables(var_to_train)



        
    num_imgs = len(os.listdir(args.train_Sharp_path))
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs', sess.graph)
    """
    if args.test_with_train:
        f = open("valid_logs.txt", 'w')
    
    epoch = 0
    step = num_imgs // args.batch_size

    # 加载一个顺序排列的图像集合列表
    blur_imgs = util.image_loader(args.train_Blur_path, args.load_X, args.load_Y)
    sharp_imgs = util.image_loader(args.train_Sharp_path, args.load_X, args.load_Y)
    deference = fp.face_points(args.train_Sharp_path, args.train_Blur_path)
    Eu_distance = fc.compute(args.train_Sharp_npy_path, args.train_Blur_npy_path)
        
    while epoch < args.max_epoch:
        #打乱数据的同时，保证数据对应，防止数据之间的相关性影响模型泛化能力
        random_index = np.random.permutation(len(blur_imgs))
        s_time = time.time()
        for k in range(step):

            blur_batch, sharp_batch = util.batch_gen(blur_imgs, sharp_imgs, args.patch_size,
                                                     args.batch_size, random_index, k)

            #计算图像对人脸之间的关键点差别
            deference_batch = fp.face_difference(deference, args.patch_size, args.batch_size, random_index, k)
            #计算特征向量之间的差别
            distance_batch = fc.face_distance(Eu_distance, args.batch_size, random_index, k)

            for t in range(args.critic_updates):
                _, D_loss = sess.run([model.D_train, model.D_loss],
                                     feed_dict={model.blur: blur_batch, model.sharp: sharp_batch, model.epoch: epoch})
                    
            _, G_loss = sess.run([model.G_train, model.G_loss],
                                 feed_dict={model.blur: blur_batch, model.sharp: sharp_batch, model.epoch: epoch, model.deference:deference_batch, model.distance:distance_batch})

                             
        e_time = time.time()

            
        if epoch % args.log_freq == 0:
            summary = sess.run(merged, feed_dict={model.blur: blur_batch, model.sharp: sharp_batch, model.deference:deference_batch, model.distance:distance_batch})
            train_writer.add_summary(summary, epoch)
            if args.test_with_train:
                test(args, model, sess, saver, f, epoch, loading=False)
            print("%d training epoch completed" % epoch)
            print("D_loss : {}, \t G_loss : {}".format(D_loss, G_loss))
            print("Elpased time : %0.4f" % (e_time - s_time))
            # print("D_loss : %0.4f, \t G_loss : %0.4f" % (D_loss, G_loss))
            # print("Elpased time : %0.4f" % (e_time - s_time))
        if (epoch) % args.model_save_freq == 0:
            saver.save(sess, './model/DeblurrGAN', global_step=epoch, write_meta_graph=False)
            
        epoch += 1
 
    saver.save(sess, './model/DeblurrGAN_last', write_meta_graph=False)
    
    if args.test_with_train:
        f.close()
        
        
def test(args, model, sess, saver, file, step=-1, loading=False):
        
    if loading:
 
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(args.pre_trained_model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(args.pre_trained_model, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")
     
    blur_img_name = sorted(os.listdir(args.test_Blur_path))
    update_img_name = sorted(os.listdir(args.train_Blur_path))
    sharp_img_name = sorted(os.listdir(args.test_Sharp_path))
    
    PSNR_list = []
    ssim_list = []
        
    blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train=False)
    sharp_imgs = util.image_loader(args.test_Sharp_path, args.load_X, args.load_Y, is_train=False)
    update_imgs = util.image_loader(args.train_Blur_path, args.load_X, args.load_Y, is_train=False)
    org_imgs = util.image_loader(args.train_Sharp_path, args.load_X, args.load_Y, is_train=False)
 
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
 
    for i, ele in enumerate(blur_imgs):
        blur = np.expand_dims(ele, axis = 0)
        sharp = np.expand_dims(sharp_imgs[i], axis = 0)
        output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim], feed_dict = {model.blur : blur, model.sharp : sharp})
        if args.save_test_result:
            output = Image.fromarray(output[0])
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))

        PSNR_list.append(psnr)
        ssim_list.append(ssim)

    if not loading and step % 10 == 9 and step != 0:

        for i, ele in enumerate(update_imgs):
            blur = np.expand_dims(ele, axis=0)
            sharp = np.expand_dims(org_imgs[i], axis=0)
            output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim],
                                          feed_dict={model.blur: blur, model.sharp: sharp})
            if args.save_test_result:
                output = Image.fromarray(output[0])
                split_name = update_img_name[i].split('.')
                # 保存训练后的图像，用于更新特征点和特征向量
                output.save(os.path.join(args.train_update_path, '%s.png' % (''.join(map(str, split_name[:-1])))))
                # 更新npy文件
        # main函数参数为需要提取特征的图片的文件夹，提取后的特征也将保存在该文件夹，请在此处更改路径
        #ef.main(args.train_update_npy_path, args.train_update_path)
 

            
    length = len(PSNR_list)


    mean_PSNR = sum(PSNR_list) / length
    mean_ssim = sum(ssim_list) / length
    

    if step == -1:
        file.write('PSNR : {} SSIM : {}' .format( mean_PSNR, mean_ssim))
        file.close()
        
    else:
        file.write("{}d-epoch step PSNR : {} SSIM : {} \n".format(step, mean_PSNR, mean_ssim))

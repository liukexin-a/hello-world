import tensorflow as tf
from DeblurGAN import DeblurGAN
from mode import *
import argparse
 
parser = argparse.ArgumentParser()
 
 
def str2bool(v):
    return v.lower() in ('True')
 

## Model specification
parser.add_argument("--n_feats", type=int, default=64)
parser.add_argument("--num_of_down_scale", type=int, default=2)
parser.add_argument("--gen_resblocks", type=int, default=9)
parser.add_argument("--discrim_blocks", type=int, default=3)
 
## Data specification 
parser.add_argument("--train_Sharp_path", type=str, default="./data/train/6/face")
parser.add_argument("--train_Blur_path", type=str, default="./data/train/6/face_blur")
parser.add_argument("--train_update_path", type=str, default="./data/train/6/6")
    #特征向量存放地址
parser.add_argument("--train_Sharp_npy_path", type=str, default="./data/train/6/face_npy")
parser.add_argument("--train_Blur_npy_path", type=str, default="./data/train/6/face_blur_npy")
parser.add_argument("--train_update_npy_path", type=str, default="./data/train/6/face_blur_npy")

parser.add_argument("--test_Sharp_path", type=str, default="./data/val/3/sharp")
parser.add_argument("--test_Blur_path", type=str, default="./data/val/3/blur")
parser.add_argument("--vgg_path", type=str, default="./vgg19.npy")
parser.add_argument("--patch_size", type=int, default=300)
parser.add_argument("--result_path", type=str, default="./result")
parser.add_argument("--model_path", type=str, default="./model")
 
## Optimization
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-6)      #正常训练1e-4
parser.add_argument("--decay_step", type=int, default=150)
parser.add_argument("--test_with_train", type=str2bool, default=True)
parser.add_argument("--save_test_result", type=str2bool, default=True)

## Training or test specification
parser.add_argument("--mode", type=str, default="test")
parser.add_argument("--critic_updates", type=int, default=5)
parser.add_argument("--augmentation", type=str2bool, default=False)
parser.add_argument("--load_X", type=int, default=300)
parser.add_argument("--load_Y", type=int, default=300)
parser.add_argument("--fine_tuning", type=str2bool, default=True)
parser.add_argument("--log_freq", type=int, default=1)
parser.add_argument("--model_save_freq", type=int, default=1)
parser.add_argument("--pre_trained_model", type=str, default="./model/")
parser.add_argument("--test_batch", type=int, default=5)
parser.add_argument("--encode", type=str, default="conv")
parser.add_argument("--update_freq", type=int, default=5)

#刚开始分配少量资源，然后按需慢慢增加GPU资源
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#sess = tf.Session(config=config)

args = parser.parse_args()
 
model = DeblurGAN(args)
model.build_graph()
 
print("Build DeblurGAN model!")
 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=None)


if args.mode == 'train':
    train(args, model, sess, saver)

elif args.mode == 'test':
    f = open("test_results.txt", 'w')
    test(args, model, sess, saver, f, step=-1, loading=True)
    f.close()
import numpy as np
import glob
import os
import csv

def compute_distance(vector1, vector2):

    Eu_distance = 0.0

    for a,b in zip(vector1,vector2):

        Eu_distance += (a-b)**2

    return Eu_distance[0]

def load_gallery(gallery_path):

    npys = sorted(os.listdir(gallery_path))

    for ele in npys:

        gallery_feature = np.load(os.path.join(gallery_path, ele))
        gallery_feature = np.reshape(gallery_feature, [2048, 1])

    return gallery_feature

def compute(gallery_path, probe_path):

    #gallery_feature = load_gallery(gallery_path)
    EuD = []
    npys = sorted(os.listdir(probe_path))
    for ele in npys:
        probe_feature = np.load(os.path.join(probe_path, ele))
        gallery_feature = np.load(os.path.join(gallery_path, ele))

        aa = compute_distance(gallery_feature, probe_feature)
        EuD.append(np.array(aa))



        #if np.sum(gallery_feature) == 0.0:
            #pass
            #print(np.shape(gallery_feature))
        #else:
            #aa = compute_distance(gallery_feature, probe_feature)
            #EuD.append(np.array(aa))

    return EuD


def face_distance(EuD, batch_size, random_index, step, augment=False):
    img_index = random_index[step * batch_size: (step + 1) * batch_size]
    all_distance = []


    for _index in img_index:


        all_distance.append(EuD[_index])

    distance_batch = []

    for i in range(len(all_distance)):

        distance_in = all_distance[i]
        if augment:
            aug = random.randrange(0, 8)
            distance_in = data_augument(distance_in, aug)

        distance_batch.append(distance_in)


    distance_batch = np.array(distance_batch[i])

    return distance_batch




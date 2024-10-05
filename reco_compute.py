import numpy as np
import glob
import os
import csv


# 计算相似度
def compute_similarity(vector1, vector2):
    #num = float(vector1.T * vector2)  # 若为行向量则 A * B.T
    #denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    #cos = num / denom  # 余弦值
    #sim = 0.5 + 0.5 * cos  # 归一化
    #return sim
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    Eu_distance = 0.0

    for a,b in zip(vector1,vector2):

        dot_product += a*b
        normA += a**2
        normB += b**2
        Eu_distance += (a-b)**2

    if normA == 0.0 or normB == 0.0:
        return None
    else:
        #返回余弦差值和特征向量的欧式距离之差平方和
        return dot_product / ((normA*normB)**0.5), Eu_distance


# 加载gallery
def load_gallery(gallery_path):
    gallery_arr = np.zeros([2048, 206])
    #获取指定目录下的所有npy文件
    gallery_lists = glob.glob(os.path.join(gallery_path, "*.npy"))
    for gallery in gallery_lists:
        #print(gallery)
        person_id = int((gallery.split('_')[-1]).split('.npy')[0])
        arr_idx = person_id - 1
        gallery_feature = np.load(gallery)
        gallery_feature = np.reshape(gallery_feature, [2048, 1])
        gallery_arr[:, arr_idx] = np.reshape(gallery_feature, 2048)

    return gallery_arr


def recognize(gallery_path, probe_path, csvfile_name):

    gallery_arr = load_gallery(gallery_path)
    probe_lists = glob.glob(os.path.join(probe_path, "*.npy"))
    reco_true = 0
    reco_false = 0
    csv_file = os.path.join(probe_path, csvfile_name)
    csv_out = open(csv_file, 'a', newline='')
    for probe in probe_lists:
        probe_id = int((probe.split('_')[-1]).split('.npy')[0])
        probe_feature = np.load(probe)
        sim = 0.00
        EuD = float('inf')
        reco_idx = 0
        for i in range(206):
            gallery_feature = gallery_arr[:, i]
            if np.sum(gallery_feature) == 0.0:
                pass
                #print(np.shape(gallery_feature))
            else:
                sim_now,EuD_now = compute_similarity(gallery_feature, probe_feature)
                if sim_now >= sim:
                    sim = sim_now
                    reco_idx = i + 1
                if EuD_now <= EuD:
                    EuD = EuD_now
        if reco_idx == probe_id:
            print(str(probe) + "  识别正确！")
            reco_true = reco_true + 1
        else:
            print(str(probe) + "  识别错误！")
            reco_false = reco_false + 1

        reco_flag = reco_idx == probe_id

        # 存入csv文件
        record = []
        record.append(probe)
        record.append(reco_idx)
        record.append(sim[0])
        record.append(reco_flag)
        csv_write = csv.writer(csv_out, dialect='excel')
        csv_write.writerow(record)

    print("Top1识别率：" + str(float(reco_true/(reco_true+reco_false))))

    # 保存识别率到csv文件
    reco_result = float(reco_true/(reco_true+reco_false))
    csv_write = csv.writer(csv_out, dialect='excel')
    reco_record = []
    reco_record.append(reco_result)
    csv_write.writerow(reco_record)


if __name__ == '__main__':
    pwd = os.getcwd()
    # 此处更改路径及csv文件名

    gallery_path = os.path.join(pwd, "Data/Temp/3/")
    probe_path = os.path.join(pwd, "Data/Temp/3/")

    csvfile_name = "result.csv"
    recognize(gallery_path, probe_path, csvfile_name)
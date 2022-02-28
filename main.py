import random
import time
import csv
from options import Options
from tqdm import tqdm
import os
from EC.GA import GA
from visulizer import concat, analyze, analyze_all, concat_best
import shutil


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    if not os.path.exists("results"):
        os.mkdir("results")
    starttime = time.time()
    random.seed(64)

    opt = Options().parse()
    # tar = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    tar = [0, 0, 0, 1, 0, 0, 0]
    iter = 1

    namelist = str.split(opt.namelist, ",")

    ana_namelist = []
    for name in tqdm(namelist):
        with open(opt.data_root + "/test_ids.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow([name])
        for i in tqdm(range(iter)):
            name = name.split(".")[0] + "_" + str(i) + "G" + str(opt.generation_count) + "L" + str(opt.life_count) + "_" + str(tar)
            ana_namelist.append(name)
            ga = GA(opt, name, tar)
            ga.run()
        break #测试的话跑一张图就行

# 一些结果可视化
    for name in ana_namelist:
        analyze(opt.results, name, iter)
        analyze_all(opt.results, name)
        concat(opt.results, name)
        concat_best(opt.results, name, 5)
    print("time cost: " + str(round(time.time() - starttime, 2)) + " seconds")


if __name__ == "__main__":
    main()

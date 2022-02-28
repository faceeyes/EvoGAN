import os
from PIL import Image
import matplotlib.pyplot as plt
import csv

def getGL(name):
    generation = int(name.split("G")[1].split("L")[0])
    life = int(name.split("L")[1].split("_")[0])
    return generation, life

def concat(path, name, size = 128, quality = 100):

    row, col = getGL(name)
    images = []
    for name2 in os.listdir(os.path.join(path, name, "imgs")):
        images.append(Image.open(os.path.join(path, name, "imgs", name2)))

    target = Image.new('RGB', (size * col, size * row))

    for i in range(len(images)):
        temp = i % row
        target.paste(images[i], (size * (int(i / row)), size * temp, size * (int(i / row) + 1), size * (temp + 1)))

    target.save(os.path.join(path, name, "concat.jpg"), quality=quality)

def analyzegen(data):
    max = 0
    min = 100
    sum = 0
    std = 0
    for d in data:
        d = float(d)
        if d > max:
            max = d
        if d < min:
            min = d
        sum += d
    avg = sum / len(data)
    for d in data:
        d = float(d)
        std += pow(d - avg, 2)
    ret = []
    ret.append(max)
    ret.append(min)
    ret.append(std)
    ret.append(avg)
    return ret

def analyze_all(path, name):


    generation, life = getGL(name)
    csv_path = os.path.join(path, name, "data.csv")
    with open(csv_path, 'r') as f1:
        reader = csv.reader(f1)
        x = []
        data = []
        next(reader)
        r = []
        for row in reader:
            r.append(row)

        cnt = 0
        for i in range(generation):
            temp = []
            for j in range(life):
                x.append(i + 1)
                data.append(float(r[cnt][4]))
                cnt = cnt + 1
        plt.clf()
        plt.plot(x, data, '.')
        plt.savefig(os.path.join(path, name, "stastic_all.pdf"), format="pdf")


def analyze(path, name, interpolation):

    generation, life = getGL(name)
    csv_path = os.path.join(path, name, "data.csv")
    with open(csv_path, 'r') as f1:
        reader = csv.reader(f1)
        lifescore = []
        genscore = []
        score = []
        next(reader)
        r = []
        for row in reader:
            r.append(row)

        i = 0
        for i1 in range(interpolation):
            for i2 in range(generation):
                for i3 in range(life):
                    lifescore.append(r[i][4])
                    i += 1
                genscore.append(lifescore)
                lifescore = []
            score.append(genscore)
            genscore = []

        max = [0 for _ in range(generation)]
        min = [0 for _ in range(generation)]
        std = [0 for _ in range(generation)]
        avg = [0 for _ in range(generation)]

        for i1 in range(generation):
            for i2 in range(interpolation):
                ret = analyzegen(score[i2][i1])
                max[i1] += ret[0]
                min[i1] += ret[1]
                std[i1] += ret[2]
                avg[i1] += ret[3]

        for j in range(generation):
            max[j] /= interpolation
            min[j] /= interpolation
            std[j] /= interpolation
            avg[j] /= interpolation

        out_csv = os.path.join(path, name,  "stastic.csv")
        with open(out_csv, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "max", "min", "std", "avg"])
            for i in range(generation):
                writer.writerow([i, max[i], min[i], std[i], avg[i]])
        x = [i for i in range(generation)]
        plt.clf()
        plt.plot(x, max, label="max")
        plt.plot(x, min, label="min")
        # plt.plot(x, std, label="std")
        plt.plot(x, avg, label="avg")
        plt.xticks(range(0, generation))
        plt.title("EvoGAN")
        plt.xlabel("generation")
        plt.ylabel("score")
        plt.legend()
        plt.savefig(os.path.join(path, name, "stastic.pdf"), format="pdf")

def concat_best(path, name, best_num, size = 128, quality = 100):

    generation, life = getGL(name)
    row = generation
    col = best_num

    csv_path = os.path.join(path, name, "data.csv")
    with open(csv_path, 'r') as f1:
        reader = csv.reader(f1)
        best = []
        next(reader)
        r = []
        for rr in reader:
            r.append(rr)

        temp = []
        i = 0
        for i1 in range(generation):
            for i2 in range(life):
                if len(temp) < best_num:
                    if len(temp) == 0 or float(r[i][4]) != temp[len(temp) - 1][2]:
                        temp.append([r[i][0], str(r[i][1]) + "_" + str(r[i][2]), float(r[i][4])])
                i += 1

            for t in temp:
                best.append(t)
            temp = []
        images = []
        for img_name in best:
            images.append(Image.open(os.path.join(path, name, "imgs", img_name[1] + ".jpg")))
            images.append(Image.open(os.path.join(path, name, "imgs", img_name[1] + ".jpg")))
        target = Image.new('RGB', (size * col, size * row))

        for i in range(len(images)):
            temp = i % row
            target.paste(images[i], (size * (int(i / row)), size * temp, size * (int(i / row) + 1), size * (temp + 1)))

        target.save(os.path.join(path, name, "concat_best.jpg"), quality=quality)

def myconcat(size = 128, quality = 100):

    path = "results"
    namelist = os.listdir(path)
    for name in namelist:
        images = []
        row, col = getGL(name)
        if row == 50 and col == 50:
            # print(name)
            for j in range(5):
                for i in range(5):
                    #print(os.path.join(path, name, "imgs", "50_" + str(i) + ".jpg"))
                    images.append(Image.open(os.path.join(path, name, "imgs", str(j * 10) + "_" + str(i) + ".jpg")))

            row = 5
            col = 5
            target = Image.new('RGB', (size * col, size * row))
            for i in range(len(images)):
                temp = i % col
                target.paste(images[i], (size * temp, size * int(i / col), size * (temp + 1), size * (int(i / col) + 1)))

            target.save((name + ".jpg"), quality=quality)

myconcat()
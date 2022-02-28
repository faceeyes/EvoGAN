import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from FER import FERtransforms as transforms
from skimage.transform import resize
from FER.FERmodels import *


class Recognizer(object):

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def recognize(self, raw_img):

        cut_size = 44

        transform_test = transforms.Compose([
            transforms.TenCrop(cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        gray = self.rgb2gray(raw_img)
        gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)

        net = VGG('VGG19')
        checkpoint = torch.load(os.path.join('FER', 'FERckpts', 'PrivateTest_model.t7'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        net.cpu()
        net.eval()

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cpu()
        with torch.no_grad():
            inputs = Variable(inputs)
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg, dim=0)
        _, predicted = torch.max(outputs_avg.data, 0)

        recscore = score.data.cpu().numpy()

        self.raw_img = raw_img
        self.score = score
        return recscore

    def save_res_img(self, img_name):

        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        plt.rcParams['figure.figsize'] = (13.5,5.5)
        axes=plt.subplot(1, 2, 1)
        plt.imshow(self.raw_img)
        plt.xlabel('Input Image', fontsize=16)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.tight_layout()

        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

        plt.subplot(1, 2, 2)
        ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
        width = 0.4       # the width of the bars: can also be len(x) sequence
        color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']
        recscore = self.score.data.cpu().numpy()
        for i in range(len(class_names)):
            plt.bar(ind[i], recscore[i], width, color=color_list[i])
        plt.title("Classification results ", fontsize=20)
        plt.xlabel(" Expression Category ", fontsize=16)
        plt.ylabel(" Classification Score ", fontsize=16)
        plt.xticks(ind, class_names, rotation=45, fontsize=14)

        plt.savefig(os.path.join('results', img_name))
        plt.close()




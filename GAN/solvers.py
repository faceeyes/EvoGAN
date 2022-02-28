from GAN.GANdata import create_dataloader
from GAN.GANmodel import create_model
import os
import torch
import numpy as np
# import face_recognition
#from GAN import cv_utils, face_utils
from PIL import Image


def create_solver(opt):
    instance = Solver()
    instance.initialize(opt)
    return instance


class Solver(object):
    """docstring for Solver"""
    def __init__(self):
        super(Solver, self).__init__()

    def initialize(self, opt):
        self.opt = opt

    def run_solver(self):
        return self.test_networks(self.opt)

    def test_networks(self, opt):
        self.init_test_setting(opt)
        return self.test_ops()

    def init_test_setting(self, opt):
        self.test_dataset = create_dataloader(opt)
        self.test_model = create_model(opt)

    def numpy2im(self, image_numpy, imtype=np.uint8):
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) / 2. + 0.5) * 255.0
        image_numpy = image_numpy.astype(imtype)
        im = Image.fromarray(image_numpy)
        return im

    def test_ops(self):
        for batch_idx, batch in enumerate(self.test_dataset):
            with torch.no_grad():

                expression = torch.unsqueeze(torch.from_numpy(self.expression), 0)
                batch['tar_aus'] = expression
                # test_batch = {'src_img': batch['src_img'], 'tar_aus': batch['tar_aus'], 'src_aus': batch['src_aus']}
                test_batch = {'src_img': batch['src_img'], 'tar_aus': batch['tar_aus']}
                self.test_model.feed_batch(test_batch)
                self.test_model.forward()

                cur_gen_faces = self.test_model.fake_img.cpu().float().numpy()

            concate_img = np.array(self.numpy2im(cur_gen_faces[0]))
            return concate_img

    def test_save_imgs(self, face, src_name, pic_name):

        face = Image.fromarray(face)
        saved_path = os.path.join(self.opt.results, src_name, "imgs", "%s.jpg" % pic_name)
        face.save(saved_path)

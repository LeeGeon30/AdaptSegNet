import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

data_dir = '/dataset/cityscapes'
data_list = './dataset/cityscapes_list/val.txt'
save_path = './result/'

ignore_label = 255
num_classes = 19
device = torch.device("cuda")
# NUM_STEPS = 500 # Number of images in the validation set.
# RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def save_cityscapes_results_for_evaluation(model):
    """Create the model and start the evaluation process."""

    if not os.path.exists(save_path):
        os.makedirs(save_path)
#     model = DeeplabMulti(num_classes=num_classes)

#     saved_state_dict = model_zoo.load_url(args.restore_from)
    
#     ### for running different versions of pytorch
#     model_dict = model.state_dict()
#     saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
#     model_dict.update(saved_state_dict)
#     ###
#     model.load_state_dict(saved_state_dict)

#     device = torch.device("cuda")
#     model = model.to(device)

    model.eval()

    testloader = data.DataLoader(cityscapesDataSet(data_dir, data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, _, name = batch
        image = image.to(device)


        output1, output2 = model(image)
        output = interp(output2).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (save_path, name))
        output_col.save('%s/%s_color.png' % (save_path, name.split('.')[0]))

if __name__ == '__main__':
    main()

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

img = Image.open('gt.gif')

img_1 = np.array(img)
img_1[img_1 >= 1] = 1

img_1 = F.sigmoid(img_1)
img_1 = img_1.resize(512,512)


img.show()
img_1.show()

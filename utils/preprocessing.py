import numpy as np
from PIL import Image
import cv2
import h5py
import matplotlib.pyplot as plt


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,2)
    assert (pred.shape[2] == 2)  # check the classes are 2
    # (Npatches,height*width)
    pred_images = np.empty((pred.shape[0], pred.shape[1]))
    if mode == "original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i, pix] = pred[i, pix, 1]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 1] >= 0.5:
                    pred_images[i, pix] = 1
                else:
                    pred_images[i, pix] = 0
    else:
        print("mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(
        pred_images, (pred_images.shape[0], 1, patch_height, patch_width))
    return pred_images


def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def load_hdf5(infile):
  with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
    return f["image"][()]


#group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    data = np.transpose(data, (0, 2, 3, 1))  # corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    img = None
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        # the image is already 0-255
        img = Image.fromarray(data.astype(np.uint8))
    else:
        # the image is between 0-1
        img = Image.fromarray((data * 255).astype(np.uint8))
    img.save(filename + '.png')
    return img


#prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert (len(masks.shape) == 4)  # 4D arrays
    assert (masks.shape[1] == 1)  # check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, 2))
    for i in range(masks.shape[0]):
        for j in range(im_h * im_w):
            if masks[i, j] == 0:
                new_masks[i, j, 0] = 1
                new_masks[i, j, 1] = 0
            else:
                new_masks[i, j, 0] = 0
                new_masks[i, j, 1] = 1
    return new_masks


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,2)
    assert (pred.shape[2] == 2)  # check the classes are 2
    # (Npatches,height*width)
    pred_images = np.empty((pred.shape[0], pred.shape[1]))
    if mode == "original":
        # for i in range(pred.shape[0]):
        #     for pix in range(pred.shape[1]):
        #         pred_images[i, pix] = pred[i, pix, 1]
        pred_images[:, :] = pred[:, :, 1]
    elif mode == "threshold":
        # for i in range(pred.shape[0]):
        #     for pix in range(pred.shape[1]):
                # if pred[i, pix, 1] >= 0.5:
                #     pred_images[i, pix] = 1
                # else:
                #     pred_images[i, pix] = 0
        pred = pred[:, :, 1]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred_images = pred
    else:
        print("mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(
        pred_images, (pred_images.shape[0], 1, patch_height, patch_width))
    return pred_images


#My pre processing (use for both training and testing!)
#<batchsize, channel, w, h>


def my_PreProc(data):
    assert(len(data.shape) == 4)
    assert (data.shape[1] == 3)  # Use the original images
    #black-white conversion

    train_imgs = kaggle_first(data)
    train_imgs = get_greenChannel(train_imgs)
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.  #reduce to 0-1 range

    #origin1:
    # train_imgs = rgb2gray(data)
    # train_imgs = dataset_normalized(train_imgs)
    # train_imgs = clahe_equalized(train_imgs)
    # train_imgs = adjust_gamma(train_imgs, 1.2)
    # train_imgs = train_imgs / 255.  #reduce to 0-1 range

    return train_imgs

#==== histogram equalization


def histo_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = cv2.equalizeHist(
            np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(
            np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
            np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs


def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:, 0, :, :] * 0.299 + \
        rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114
    bn_imgs = np.reshape(
        bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs


def kaggle_first(data):
    for i in range(data.shape[0]):
        img = data[i, :, :, :]
        img = np.transpose(img,(1,2,0))

        # convert to lab
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)

        #-----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(img)

        #-----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))

        #-----Converting image from LAB Color model to RGB model--------------------
        s = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        s = np.transpose(s, (2, 0, 1))
        data[i, :, :, :] = s
    return data

def get_greenChannel(data):
    data = data[:, 1, :, :]
    return np.expand_dims(data, 1)


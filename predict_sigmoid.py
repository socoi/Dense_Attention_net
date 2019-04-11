import os
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import torch.nn.functional as F
from PIL import Image
from unet import UNet
from utils.extract_patches import my_PreProc, paint_border, extract_ordered, recompone, kill_border, recompone_overlap, pred_only_FOV, get_data_testing, get_data_testing_overlap
from utils.preprocessing import mask_to_image, plot_img_and_mask, pred_to_imgs, load_hdf5, visualize, group_images
from torch.utils.data import Dataset, DataLoader
from sklearn import cluster


dir_checkpoint = os.getcwd() + '\\h5_data\\'
model = '11.pth'
Imgs_to_test = 20 # 20 test image
mask_threshold = 0.3
batch_size = 128
N_visual = 2 # visualzation images rows



# define pyplot parameters
import matplotlib.pylab as pylab
params = {'legend.fontsize': 15,
          'axes.labelsize': 15,
          'axes.titlesize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}
pylab.rcParams.update(params)

patch_height = 48
patch_width = 48
stride_height = 5
stride_width = 5
DRIVE_test_imgs_original = dir_checkpoint + 'DRIVE_dataset_imgs_test.hdf5'
DRIVE_test_groudTruth = dir_checkpoint + 'DRIVE_dataset_groundTruth_test.hdf5'
DRIVE_test_borderMask = dir_checkpoint + 'DRIVE_dataset_borderMasks_test.hdf5'



class TrainDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs):
        self.imgs = patches_imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx,...]).float()




def predict_img(net,
                mask_threshold,
                use_gpu):

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve


    # CAM
    output_cam = []

    # patch
    preds = []

    def hook_feature(module, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
        features_blobs.append(output.data.cpu().numpy())

    #============ Load the data and divide in patches(average mode)
    # in this mode,height and mask is new
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        DRIVE_test_imgs_original=DRIVE_test_imgs_original,
        DRIVE_test_groudTruth=DRIVE_test_groudTruth,
        Imgs_to_test=Imgs_to_test,
        patch_height=patch_height,
        patch_width=patch_width,
        stride_height=stride_height,
        stride_width=stride_width
    )

    test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
    test_imgs_groudTruth = load_hdf5(DRIVE_test_groudTruth)
    test_imgs_bordermask = load_hdf5(DRIVE_test_borderMask)

    test_set = TrainDataset(patches_imgs_test)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    # create CAM(hook)
    net._modules.get('up4').register_forward_hook(hook_feature)

    for batch_idx, inputs in enumerate((test_loader)):

        # hook the feature extractor
        features_blobs = []

        inputs = inputs.cuda()
        outputs = net(inputs)
        outputs = outputs.data.cpu().numpy()
        preds.append(outputs)


        # get the weight
        params = list(net.parameters())
        weight = np.squeeze(params[-2].data.cpu().numpy()) #64 channel
        batch_weight = np.empty((outputs.shape[0], 1, weight.shape[0]))
        batch_weight[:, :, :] = weight

        # cam_feature_map
        bz, nc, h, w = features_blobs[0].shape


        # multi-dimension  multi
        feature_batch = features_blobs[0].reshape((bz, nc, h * w))
        cam = np.einsum('ijk,ikl->ijl', batch_weight, feature_batch)
        cam = cam.reshape(outputs.shape[0], 1, h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        output_cam.append(cam_img)

    predictions = np.concatenate(preds, axis=0)
    get_CAM = np.concatenate(output_cam, axis=0)

    del preds
    del output_cam

    # np.save('predictions',predictions)
    # np.save('get_CAM', get_CAM)

    # predictions = np.load('predictions.npy')
    # get_CAM = np.load('get_CAM.npy')



    print("Predictions finished")
    #===== Convert the prediction arrays in corresponding images
    #pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
    #predictions[predictions >= mask_threshold] = 1
    #predictions[predictions < mask_threshold] = 0

    #========== Elaborate and visualize the predicted images ====================
    pred_imgs = recompone_overlap(
        predictions, new_height, new_width, stride_height, stride_width)  # predictions
    pred_CAM = recompone_overlap(get_CAM, new_height, new_width, stride_height, stride_width)
    # pred_fusion = recompone_overlap(fusion_predict, new_height, new_width, stride_height, stride_width)

    pred_imgs = pred_imgs[0:Imgs_to_test]
    pred_CAM = pred_CAM[0:Imgs_to_test]

    gtruth_masks = masks_test  # ground truth masks


    orig_imgs = my_PreProc(
        test_imgs_orig[0:pred_imgs.shape[0], :, :, :])  # originals

    pred_imgs = pred_imgs[:, :, 0:masks_test.shape[2], 0:masks_test.shape[3]]
    pred_CAM = pred_CAM[:, :, 0:masks_test.shape[2], 0:masks_test.shape[3]]
    #pred_fusion = pred_fusion[:, :, 0:masks_test.shape[2], 0:masks_test.shape[3]]

    pred_Final = np.empty((Imgs_to_test, 1, masks_test.shape[2], masks_test.shape[3]))

    for i in range(0, Imgs_to_test):
        # k-means
        m = pred_CAM[i].reshape(masks_test.shape[2] * masks_test.shape[3], -1)
        kmeans = cluster.KMeans(n_clusters=2, max_iter=5,
                                precompute_distances=True)
        pred_fusion = kmeans.fit_predict(m)
        pred_fusion = pred_fusion.reshape(
            1, 1, masks_test.shape[2], masks_test.shape[3])

        # means
        pred_fusion = (pred_fusion + np.expand_dims(pred_imgs[i], axis=0)) / 2


        pred_Final[i, :, :, :] = pred_fusion[0]


     # feature fusion
     # need original image
    # t1 = np.transpose(pred_imgs[0] * 255, (1, 2, 0)).squeeze().astype(np.uint8)
    # t2 = np.transpose(pred_CAM[0] * 255, (1, 2, 0)).squeeze().astype(np.uint8)
    # t3 = gtruth_masks.squeeze().squeeze().astype(np.uint8)
    # t4 = (masks_test.shape[2] // 2, masks_test.shape[3] // 2)

 
    # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
    # pred_imgs = kill_border(pred_imgs, test_imgs_bordermask)  #DRIVE MASK
    # pred_CAM = kill_border(pred_CAM, test_imgs_bordermask)
    # pred_Final = kill_border(pred_Final, test_imgs_bordermask)
    #   #only for visualization

    # #pred_imgs[pred_imgs >= mask_threshold] = 1
    # #pred_imgs[pred_imgs < mask_threshold] = 0


    # visualize(group_images(pred_imgs, N_visual), 'a').show()
    # visualize(group_images(pred_CAM, N_visual), 'b').show()
    # visualize(group_images(pred_fusion, N_visual), 'c').show()
    #visualize(group_images(gtruth_masks, N_visual), 'c').show()

    #====== Evaluate the results
    print("\n\n========  Evaluate the results =======================")
    #predictions only inside the FOV
    # returns data only inside the FOV
    y_scores1, y_true1 = pred_only_FOV(pred_imgs, gtruth_masks, test_imgs_bordermask)
    y_scores2, y_true2 = pred_only_FOV(pred_CAM, gtruth_masks, test_imgs_bordermask)
    y_scores3, y_true3 = pred_only_FOV(pred_Final, gtruth_masks, test_imgs_bordermask)


    #Area under the ROC curve
    fpr1, tpr1, thresholds = roc_curve((y_true1), y_scores1)
    fpr2, tpr2, thresholds = roc_curve((y_true2), y_scores2)
    fpr3, tpr3, thresholds = roc_curve((y_true3), y_scores3)
    AUC_ROC1 = roc_auc_score(y_true1, y_scores1)
    AUC_ROC2 = roc_auc_score(y_true2, y_scores2)
    AUC_ROC3 = roc_auc_score(y_true3, y_scores3)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    #print("\nArea under the ROC curve: " + str(AUC_ROC1))
    roc_curve = plt.figure()
    plt.plot(fpr1, tpr1, 'b-', label='U-net (AUC = %0.4f)' % AUC_ROC1)
    plt.plot(fpr2, tpr2, 'r-', label='CAM  (AUC = %0.4f)' % AUC_ROC2)
    plt.plot(fpr3, tpr3, 'g-', label='Fusion (AUC = %0.4f)' % AUC_ROC3)

    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":


    net = UNet(n_channels=1, n_classes=1)
    net = net.eval()

    net.cuda()
    net.load_state_dict(torch.load(os.getcwd() + '\\h5_data\\' + model))
    print("Model loaded !")


    mask = predict_img(net=net,
                       mask_threshold=mask_threshold,
                       use_gpu=True)

    #plot_img_and_mask(img, mask)

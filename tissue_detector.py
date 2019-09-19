import numpy as np
from PIL import Image
import cv2 as cv
from sklearn.naive_bayes import GaussianNB

def get_tissue_mask(I, luminosity_threshold=0.8):
    """
    Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
    Typically we use to identify tissue in the image and exclude the bright white background.

    :param I: RGB uint 8 image numpy array.
    :param luminosity_threshold: Luminosity threshold.
    :return: Binary mask.
    """
    assert is_uint8_image(I), "Image should be RGB uint8."
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold

    # Check it's not empty
    if mask.sum() == 0:
        raise TissueMaskException("Empty tissue mask computed")
    return mask
  
def train_GNB(pixel_rgb_tsv_file, dim):
    tsv_cols = np.loadtxt(pixel_rgb_tsv_file, delimiter="\t", skiprows=1)
    bkg_train_data = tsv_cols[:, 0:feature_dim + 1]
    gnb_bkg = GaussianNB()
    gnb_bkg.fit(bkg_train_data[:, 1:], bkg_train_data[:, 0])
    return gnb_bkg


def gnb_get_foreground(WSI_thumb_img, GNB_model, threshold=0.5):
    marked_thumbnail = np.array(WSI_thumb_img)
    cal = GNB_model.predict_proba(marked_thumbnail.reshape(-1, 3))
    cal = cal.reshape(marked_thumbnail.shape[0], marked_thumbnail.shape[1], 2)
    binary_img_array = cal[:, :, 1] < threshold
    return binary_img_array

if __name__=="__main__":
    WSI_thumb_img = Image.open("test.jpg")
    pixel_rgb_tsv_file = "bkg_others.tsv"
    GNB_model = train_GNB(pixel_rgb_tsv_file, 3)
    gnb_mask = gnb_get_foreground(WSI_thumb_img, GNB_model, threshold=0.5)


    

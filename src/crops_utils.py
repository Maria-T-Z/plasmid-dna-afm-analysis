import os
import cv2
import warnings
import numpy as np
from PIL import Image
from skimage.filters import frangi, threshold_li
from skimage.measure import label
warnings.filterwarnings("ignore")



def segment_plasmid_frangi(img_8bit: np.ndarray) -> np.ndarray:
    """
    Segmentation pipeline
    """
    #NLM
    h_nlm = 10
    img_nlm = cv2.fastNlMeansDenoising(img_8bit, None, h=h_nlm, templateWindowSize=7, searchWindowSize=21)

    #Frangi
    s_range = np.arange(1, 4, 1)  # Sigmas: 1, 2, 3
    a, b, g = 0.5, 0.5, 15
    out_frangi = frangi(img_nlm, sigmas=s_range, black_ridges=False, alpha=a, beta=b, gamma=g)

    #LI and largest elemet
    if out_frangi.max() > out_frangi.min():
        thresh = threshold_li(out_frangi)
        mask_raw = out_frangi > thresh

        labeled_mask = label(mask_raw)
        if labeled_mask.max() > 0:
            largest_label = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
            mask_final = (labeled_mask == largest_label)
        else:
            mask_final = mask_raw
    else:
        mask_final = np.zeros_like(out_frangi, dtype=bool)

    return (mask_final * 255).astype(np.uint8)



def process_and_save_crops(npy_path: str, png_path: str, boxes: list, output_dir: str, base_name: str) -> int:
    """
    Creating crops - based on YOLO. For .png and .npy data. Saving that data.
    Args:
        npy_path (str): Path to the flattened physical matrix.
        png_path (str): Path to the normalized 8-bit image.
        boxes (list): List of coordinates [x1, y1, x2, y2] from YOLO.
        output_dir (str): Root directory to save the crops.
        base_name (str): Original filename base (to name the crops).

    Returns:
        int: Number of valid crops successfully extracted, segmented, and saved.
    """
    # Load physical and visual data
    data_matrix = np.load(npy_path)
    img_pil = Image.open(png_path)
    img_array = np.array(img_pil)

    npy_out_dir = os.path.join(output_dir, "npy")
    png_out_dir = os.path.join(output_dir, "png")
    mask_out_dir = os.path.join(output_dir, "masks")

    os.makedirs(npy_out_dir, exist_ok=True)
    os.makedirs(png_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    crops_saved = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        if (x2 - x1) < 5 or (y2 - y1) < 5: #ignoring areas smaller than 5x5
            continue

        crop_npy = data_matrix[y1:y2, x1:x2]
        crop_png = img_array[y1:y2, x1:x2]

        crop_mask = segment_plasmid_frangi(crop_png)

        crop_filename = f"{base_name}_crop_{i:03d}"

        np.save(os.path.join(npy_out_dir, f"{crop_filename}.npy"), crop_npy)
        Image.fromarray(crop_png).save(os.path.join(png_out_dir, f"{crop_filename}.png"))
        Image.fromarray(crop_mask).save(os.path.join(mask_out_dir, f"{crop_filename}_mask.png"))

        crops_saved += 1

    return crops_saved
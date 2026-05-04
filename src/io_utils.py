import numpy as np
from spmpy import SPMFile
import os
from PIL import Image


"HELPER FUNCTIONS"

"part 1 - flattening"
def fit_polynomial_surface(x, y, z, order=2):
    """ Fits a 1st or 2nd order polynomial surface to 3D data points.
    Args:
        x (np.ndarray): X-coordinates grid.
        y (np.ndarray): Y-coordinates grid.
        z (np.ndarray): Z-height values.
        order (int, optional): Polynomial order (1 for flat plane, 2 for curved surface). Defaults to 2.
    Returns:
        np.ndarray: The fitted polynomial surface evaluated at the given (x, y) grid."""
    X = x.ravel()
    Y = y.ravel()
    Z = z.ravel()

    if order == 1:
        A = np.column_stack((X, Y, np.ones_like(X)))
    elif order == 2:
        A = np.column_stack((X, Y, X**2, X*Y, Y**2, np.ones_like(X)))
    else:
        raise ValueError("only order=1 or 2!!!")
    #metoda najmniejsyzch kwadratów
    C, *_ = np.linalg.lstsq(A, Z, rcond=None)
    #dopasowana powierzchnia
    if order == 1:
        surface = C[0]*x + C[1]*y + C[2]
    else:
        surface = (C[0]*x + C[1]*y + C[2]*x**2 + C[3]*x*y + C[4]*y**2 + C[5])
    return surface

def flatten_base_advanced(data, order=2, mask_percentile=90, iterations=3):
    """
        Iteratively flattens an AFM image background, inspired by Gwyddion's flattening algorithm.
        Fits a polynomial to the background while dynamically masking out large features
        (like plasmids) to prevent artificial trenches from forming around tall objects.
        The background mask is refined over a set number of iterations.
        Args:
            data (np.ndarray): 2D array representing the original AFM height map.
            order (int, optional): Order of the polynomial surface to fit. Defaults to 2.
            mask_percentile (float, optional): Percentile of data to consider as the background.
                For example, 90.0 means the top 10% highest points are ignored during fitting. Defaults to 90.0.
            iterations (int, optional): Number of masking refinement iterations. Defaults to 3.
        Returns:
            np.ndarray: The flattened data (residuals) with the background leveled near zero.
        """
    #jak to z gwyddiona
    #dopasowuje wielomian do tła, maskuje duże obiekty + iteracja maski

    #wspolrzedne, maska
    ny, nx = data.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    current = data.copy()
    mask = np.ones_like(data, dtype=bool)

    for it in range(iterations):
        #obliczenia na siatece współrzędnych
        surface_full = fit_polynomial_surface(X, Y, data, order=order)
        #reszty
        residuals = data - surface_full
        #próg maskowania
        low = np.percentile(residuals[mask], 100-mask_percentile)
        high = np.percentile(residuals[mask], mask_percentile)
        #aktualizacja maski i current
        mask = (residuals >= low) & (residuals <= high)
        current = residuals
    return current



def step_line_correction_advanced(data, jump_threshold=None, min_segment_length=10):
    #korekcja skoków
    """
    inspirowane gwyddion Step Line Correction.
    data: 2D ndarray
    jump_threshold: jeśli None, wyznaczy automatyczniee na podstawie odchylenia
    min_segment_length: min dł segmentu między skokami
    """
    corrected = np.zeros_like(data, dtype=float)
    nrows, ncols = data.shape

    #jeśli próg nie jest podany – podstawie odchylenia
    if jump_threshold is None:
        diffs = np.diff(data, axis=1).ravel()
        jump_threshold = 3 * np.std(diffs)
        print(f"[INFO] Auto jump_threshold = {jump_threshold:.4f}")

    for i in range(nrows):
        row = data[i, :]
        #różnice miedzy sąsiednimi pixelami
        d = np.diff(row)
        #id skoków (gdzie różnica > próg)
        jump_indices = np.where(np.abs(d) > jump_threshold)[0] + 1

        #segmenty in rows
        segments = [0] + jump_indices.tolist() + [ncols]
        new_row = np.zeros_like(row)
        for s in range(len(segments) - 1):
            start = segments[s]
            end = segments[s + 1]
            if end - start < min_segment_length:
                continue
            segment = row[start:end]
            seg_mean = np.mean(segment)
            new_row[start:end] = segment - seg_mean
        corrected[i, :] = new_row
    return corrected

def standardize_z_units(data):

    z_range = np.ptp(data)
    # If amplitude is less than 0.1 nm, assume it's actually in micrometers.
    if z_range < 0.1:
        return data * 1000.0
    return data








"MAIN FUNCTIONS"


def load_spm(path):
    """Loading .spm files (AFM format), getting metadata and Height channel"""
    spm_file = SPMFile(path)
    height_q = spm_file.images['Height'] #height channel
    height_raw = np.array(height_q.magnitude, dtype=float)
    height_raw = standardize_z_units(height_raw) #if the units are wrong

    #getting scale -> only for square format?
    extent = height_q.extent
    width_nm = extent[1]
    px_count = height_raw.shape[0]
    scale = width_nm / px_count

    return height_raw, scale, spm_file

def flatten_and_correct(data):
    """Img correction using previous helper functions"""
    flattened = flatten_base_advanced(data, order=2, mask_percentile=85, iterations=5)
    corrected = step_line_correction_advanced(flattened)
    return corrected


def save_preprocessed(data, scale, filename, output_dir):
    """Saving 2 formats: .png and .npy + scale"""
    base = os.path.splitext(os.path.basename(filename))[0]
    os.makedirs(os.path.join(output_dir, "heights"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    #height map
    np.save(os.path.join(output_dir, "heights", f"{base}.npy"), data)

    #8-bit png
    img_rescaled = (data - data.min()) / (data.max() - data.min()) * 255
    img_uint8 = img_rescaled.astype(np.uint8)
    Image.fromarray(img_uint8).save(os.path.join(output_dir, "images", f"{base}.png"))

    #scale
    with open(os.path.join(output_dir, "metadata", f"{base}_scale.txt"), "w") as f:
        f.write(str(scale))
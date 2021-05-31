from collections import namedtuple
import concurrent.futures
import rasterio
import numpy as np
import scipy.ndimage as ndimage
from tqdm import tqdm

Strip = namedtuple('Strip', ['start', 'end', 'margin_top', 'margin_bottom'])

def compute_margin_for_level(num_outer_iterations: int,
                             num_inner_iterations: int,
                             uniform_filter_size: int) -> int :
    return num_outer_iterations * num_inner_iterations * uniform_filter_size

def retrieve_dsm_resolution(dsm_dataset: rasterio.DatasetReader) -> float:
    """ """
    # We assume that resolution is the same wrt to both image axis
    print(dsm_dataset.transform)
    return 0.0

def get_max_pyramid_level(max_object_size_pixels: int) -> int :
    """ 
        Given the max size of an object on the ground,
        this methods compute the max level of the pyramid
        for drap cloth algorithm
    """
    power = 0
    while 2**power < max_object_size_pixels:
            power+=1
    
    # Take the closest power to the max object size
    if abs(2**(power-1) - max_object_size_pixels) <  abs(2**(power) - max_object_size_pixels):
    power -= 1

    return power

def downsample(buffer: np.ndarray) -> np.ndarray:
    """
        Simple 2X downsampling, take every other pixel
    """
    return buffer[::2, ::2]

def upsample(buffer: np.ndarray, 
             out: np.ndarray):
    """
        Simple 2X upsampling, duplicate pixels
    """
    # Adjust the slicing for odd row count
    if out.shape[0] % 2 == 1:
        s0 = numpy.s_[:-1]
    else:
        s0 = numpy.s_[:]

    # Adjust the slicing for odd column count
    if out.shape[1] % 2 == 1:
        s1 = numpy.s_[:-1]
    else:
        s1 = numpy.s_[:]

    # copy in duplicate values for blocks of 2x2 pixels
    out[::2, ::2] = buffer
    out[1::2, ::2] = buffer[s0, :]
    out[::2, 1::2] = buffer[:, s1]
    out[1::2, 1::2] = buffer[s0, s1]

def build_pyramid(dsm_dataset: rasterio.DatasetReader,
                  nb_levels: int) -> list :
    """
        Given the number of levels, this method builds the
        dsm pyramid.
    """
    dsm_pyramid = []

    dsm_pyramid.append(dsm_dataset.read(1))
    for l in range(1, nb_levels):
        dsm_pyramid.append(downsample(dsm_pyramid[l-1]))

    return dsm_pyramid

def prevent_unhook_from_hills(dsm_pyramid: list,
                              prevent_unhook_iter: int) -> np.ndarray:
    """
        This first step is introduced for preventing the drap
        to unhook from hills. To do that, the DTM is initialized
        to the DSM with the highest dezoom and then only ninner iterations
        are applied to smooth the dtm
    """
    # Dtm is initialized at the most dezoomed dsm
    dtm = np.copy(dsm_pyramid[-1])

    for i in range(prevent_unhook_iter):
        dtm = ndimage.uniform_filter(dtm, size=3)
    
    return dtm

def sequential_drap_cloth(dtm: np.ndarray,
                          dsm: np.ndarray,
                          n_outer_iter: int,
                          n_inner_iter: int,
                          step: float):
    for i in range(n_outer_iter):
        dtm += step
        for i in range(n_inner_iter):
            # handle DSM intersections, snap back to below DSM
            np.minimum(dtm, dsm, out=dtm)
            # apply spring tension forces (blur the DTM)
            dtm = ndimage.uniform_filter(dtm, size=3)

        # Final check intersection, snap back to below DSM
        np.minimum(dtm, dsm, out=dtm)

def compute_margin_top(n: int,
                       margin: int,
                       start: int) -> int:
    if n > 0:
        if (start - margin) < 0:
            return start
        else:
            return margin
    else:
        # There is no top margin for the first strip
        return 0

def compute_margin_bottom(n: int,
                          nb_cpus: int,
                          margin: int,
                          end: int,
                          buffer_height: int) -> int:
    if n < nb_cpus - 1:
        if end + margin > buffer_height - 1:
            return buffer_height - 1 - end
        else:
            return margin
    else:
        # There is no bottom margin for the last strip
        return 0

def compute_strips(dtm: np.ndarray,
                   nb_cpus: int,
                   margin: int) -> list :

    strip_height = dtm.shape[0] // nb_cpus
    remainder_strip = dtm.shape[0] % nb_cpus    
    strips = []
    
    for n in range(nb_cpus):
        start = n*strip_height
        end = (n+1)*strip_height-1 if n < nb_cpus - 1 else (n+1)*strip_height-1 + remainder_strip
        margin_top = compute_margin_top(n, margin, start)
        margin_bottom = compute_margin_bottom(n, nb_cpus, margin, end, dtm.shape[0])
        strips.append(Strip(start, end, margin_top, margin_bottom))
    
    return strips

def chunk_drap_cloth(dtm_path: str,
                     dsm_path: str,
                     strip: Strip,
                     width: int,
                     n_outer_iter: int,
                     n_inner_iter: int,
                     step: float):

    start_y = strip.start - strip.margin_top
    size_y = strip.end + strip.margin_bottom - start_y + 1 
    wd = Window(0, start_y, width, size_y)
    dtm = rasterio.open(dtm_path).read(window=wd)
    dsm = rasterio.open(dsm_path).read(window=wd)
    

    for i in range(n_outer_iter):
        dtm += step
        for i in range(n_inner_iter):
            # handle DSM intersections, snap back to below DSM
            np.minimum(dtm, dsm, out=dtm)
            # apply spring tension forces (blur the DTM)
            dtm = ndimage.uniform_filter(dtm, size=3)

        # Final check intersection, snap back to below DSM
        np.minimum(dtm, dsm, out=dtm)
    
    return ( dtm[strip.margin_bottom:strip.end+strip.margin_bottom+1, :], strip )

def parallel_drap_cloth(ref_dataset: rasterio.DatasetReader,
                        dtm: np.ndarray,
                        dsm: np.ndarray,
                        n_outer_iter: int,
                        n_inner_iter: int,
                        step: int,
                        nb_cpus: int,
                        output_directory: str):

    # Compute stable margin
    margin = compute_margin_for_level(n_outer_iter, num_inner_iterations, 3)

    # Compute list of strips to run in //
    strips = compute_strips(dtm, nb_cpus, margin)

    # Write temporary to disk and delete to avoid memory redundancy
    profile = ref_dataset.profile
    profile["height"] = dtm.shape[0]
    profile["width"] = dtm.shape[1]
    
    in_tmp_dtm_path = output_directory + "/tmp_dtm.tif"
    in_tmp_dsm_path = output_directory + "/tmp_dsm.tif"

    with rasterio.open(in_tmp_dtm_path, "w", **profile) as dst:
        dst.write(1, dtm)

    with rasterio.open(in_tmp_dsm_path, "w", **profile) as dst:
        dst.write(1, dsm)
    
    dtm = None
    dsm = None

    out_dtm_dataset = rasterio.open(output_directory + "/dtm.tif", "w", **profile)

    # Run in drap cloth in //
    with concurrent.futures.ProcessPoolExecutor(max_workers=nb_cpus) as executor:
        futures = {executor.submit(chunk_drap_cloth,
                                   in_tmp_dtm_path,
                                   int_tmp_dsm_path,
                                   strip,
                                   profile["width"],
                                   n_outer_iter,
                                   n_inner_iter,
                                   step) for strip in strips}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Drap Cloth execution"):
            chunk_dtm, strip = future.result()
            wd = Window(0, strip.start, profile["width"], strip.end - strip.start + 1)
            out_dtm_dataset.write(chunk_dtm, window=wd)
        
    out_dtm_dataset.close()

    # Now we can load the new dtm and the dsm
    dtm = rasterio.open(output_directory + "/dtm.tif").read(1)
    dsm = rasterio.open(in_tmp_dsm_path).read(1)
#
# Large scale implementation of CNES strategy for drap cloth
# strip parallel execution when strip height is lower than 
# dsm to process.
#
def dtm_extraction(in_dsm_path: str,
                   out_dtm_directory: str,
                   min_height: int,
                   nb_cpus: int,
                   max_object_size: int,
                   prevent_unhook_iter: int,
                   num_outer_iterations: int,
                   num_inner_iterations: int):
    """
        This methods executes the extraction of the modified drap cloth
        algorithm using multi processing.

        Args:
            in_dsm_path: path to the dsm file
            out_dtm_directory: ouput directory path where dtm will be stored (dtm.tif) and where temporary rasters will be stored and removed.
            min_height: minimum image height to process in //
            nb_cpus: maximum number of cpus to use in //
            max_object_size: max size of an over-ground object
            prevent_unhook_iter: Retrieve the number of inner iterations for prenventing from unhooking from hills.
            num_outer_iterations: Number of iterations for drap fall by gravity
            num_inner_iterations: Number of iterations to smooth the drap after a gravity fall
        Returns:

        Raise:
    """

    # Open the dsm dataset
    in_dsm_dataset = rasterio.open(in_dsm_path)

    # Retrieve dsm resolution
    dsm_res = retrieve_dsm_resolution(in_dsm_dataset)

    # Determine max object size in pixels
    max_object_size_pixels = max_object_size / dsm_res

    # Determine the dezoom factor wrt to max size of an object
    # on the ground.
    nb_levels = get_max_pyramid_level(max_object_size_pixels) + 1

    # Build the dsm pyramid
    dsm_pyramid = build_pyramid(dsm_dataset, nb_levels)

    # Prevent from unhooking from hills
    dtm = prevent_unhook_from_hills(dsm_pyramid, prevent_unhook_iter)

    # Init classical parameters of drap cloth
    min_alt = np.min(dsm_pyramid[0])
    max_alt = np.max(dsm_pyramid[0])
    step = (max_alt - min_alt) / num_outer_iterations
    max_level = nb_levels - 1
    level = max_level
    n_outer_iter = num_outer_iterations

    # On applique le multi-scale de len(dsmPyramid) - 2 à 0
    j = len(dsm_pyramid) - 2
    while j > 0:

        # Upsample dtm to lower level
        dtmNext = np.copy(dsm_pyramid[j])
        drapClothHandler.upsample(dtm, dtmNext)
        dtm = dtmNext

        if dtm.shape[0] >= min_height:
            # Run in //
            parallel_drap_cloth(ref_dataset = in_dsm_dataset,
                                dtm = dtm,
                                dsm = dsm_pyramid[j],
                                n_outer_iter = n_outer_iter,    
                                n_inner_iter = num_inner_iterations,
                                step=step
                                nb_cpus = nb_cpus,
                                output_directory=out_dtm_directory)
        else:
            # Sequential run
            sequential_drap_cloth(dtm=dtm,
                                  dsm=dsm_pyramid[j],
                                  n_outer_iter = n_outer_iter,
                                  n_inner_iter = num_inner_iterations,
                                  step=step)
        
        # Decrease step
        step = step / (2 * 2 ** (max_level - level))
        # Decrease the number of iterations as well
        n_outer_iter = max(1, int(numOuterIterationsStep2 / (2 ** (max_level - level))))
        # Decrease the current level
        level-=1
        j+=1


    in_dsm_dataset.close()

if __name__ == "__main__":
    print("Hello")

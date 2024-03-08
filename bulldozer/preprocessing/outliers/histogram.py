import numpy as np
import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe
import bulldozer.preprocessing.outliers.stats as pystats
import bulldozer.preprocessing.outliers.hist as pyhist

def minmax_filter(input_buffers : list, 
                  input_profiles: list, 
                  filter_parameters: dict):
    
    stats_filter = pystats.PyStats()
    if filter_parameters['is_invalid_mask']:
        return stats_filter.computeStats(input_buffers[0][0], input_buffers[1][0], float(input_profiles[0]["nodata"]))
    else:
        return stats_filter.computeStats(input_buffers[0][0], np.ones(input_buffers[0][0].shape).astype(np.uint8), float(input_profiles[0]["nodata"]))

def minmax_concatenate(output_scalars, chunk_output_scalars, tile):
    output_scalars[0] = min(output_scalars[0], chunk_output_scalars[0] )
    output_scalars[1] = max(output_scalars[1], chunk_output_scalars[1] )

def configure_histogram(min_z: float, 
                        max_z: float, 
                        dsm_z_precision: float) -> tuple:
    """
        Given valid min_z and max_z, this method computes the number of bins and
        the bin width.
    """
    bin_width: float = 2 * dsm_z_precision
    nb_bins: int = int ( (max_z - min_z) / bin_width  ) + 1
    return (nb_bins, bin_width)

def compute_hist(input_buffers : list, 
                 input_profiles: list, 
                 filter_parameters: dict):
    """ """
    hist_filter = pyhist.PyHist()
    if filter_parameters['is_invalid_mask']:
        return hist_filter.computeHist(input_buffers[0][0], input_buffers[1][0],
                                   float(filter_parameters["dsm_min_z"]),
                                   int(filter_parameters["nb_bins"]),
                                   float(filter_parameters["bin_width"]),
                                   float(input_profiles[0]["nodata"]))
    else:
        return hist_filter.computeHist(input_buffers[0][0], np.ones(input_buffers[0][0].shape).astype(np.uint8),
                                   float(filter_parameters["dsm_min_z"]),
                                   int(filter_parameters["nb_bins"]),
                                   float(filter_parameters["bin_width"]),
                                   float(input_profiles[0]["nodata"]))

def concatenate_hist(output_scalars, 
                     chunk_output_scalars, 
                     tile):
    output_scalars[0] += chunk_output_scalars[0]

def compute_robust_z_interval(hist: np.ndarray,
                              dsm_min_z: float,
                              bin_width: float,
                              dsm_z_precision: float) -> float:
    indices = np.argwhere(hist >= np.mean(hist))

    return indices[0][0] * bin_width + dsm_min_z - dsm_z_precision, indices[0][-1] * bin_width + dsm_min_z + dsm_z_precision

def compute_uncertain_mask(input_buffers : list, 
                           input_profiles: list, 
                           filter_parameters: dict) -> np.ndarray:
    """ """
    return np.where( np.logical_or(input_buffers[0] < filter_parameters["min_z"], 
                                   input_buffers[0] > filter_parameters["max_z"]), 1, 0).astype(np.uint8)

def uncertain_profile(input_profiles: list,
                      params: dict) -> dict:
    input_profiles[0]['dtype'] = np.uint8
    input_profiles[0]['nodata'] = None
    return input_profiles[0]


def run(input_dsm_key: str,
        eomanager: eom.EOContextManager,
        dsm_z_precision: float,
        input_invalid_key : str = None) -> dict :

    """ """
    min_max_parameters = {}
    min_max_parameters['is_invalid_mask'] = True if input_invalid_key is not None else False
    # Compute valid min and max heights (taking into account nodata values)
    if(min_max_parameters['is_invalid_mask']):
        [dsm_min, dsm_max] = eoexe.n_images_to_m_scalars(inputs = [input_dsm_key, input_invalid_key],
                                                     filter_parameters=min_max_parameters,
                                                     image_filter = minmax_filter,
                                                     nb_output_scalars = 2,
                                                     concatenate_filter = minmax_concatenate,
                                                     context_manager = eomanager,
                                                     filter_desc= "Min/Max value processing...")
    else:
        [dsm_min, dsm_max] = eoexe.n_images_to_m_scalars(inputs = [input_dsm_key],
                                                     filter_parameters=min_max_parameters,
                                                     image_filter = minmax_filter,
                                                     nb_output_scalars = 2,
                                                     concatenate_filter = minmax_concatenate,
                                                     context_manager = eomanager,
                                                     filter_desc= "Min/Max value processing...")
    print("min: ", dsm_min, "\nmax:", dsm_max)
    #  Histogram computation
    nb_bins, bin_width = configure_histogram(min_z = dsm_min, 
                                             max_z = dsm_max, 
                                             dsm_z_precision = dsm_z_precision)

    # Compute the number of bins and bin width
    nb_bins, bin_width = configure_histogram(dsm_min, dsm_max, dsm_z_precision)

    hist_params: dict = {
        "dsm_min_z": dsm_min,
        "nb_bins": nb_bins,
        "bin_width": bin_width,
        "is_invalid_mask" : True if input_invalid_key is not None else False
    }
    if(min_max_parameters['is_invalid_mask']):
        hist = eoexe.n_images_to_m_scalars(inputs = [input_dsm_key, input_invalid_key],
                                            filter_parameters = hist_params,
                                        image_filter = compute_hist,
                                        nb_output_scalars = 1,
                                        output_scalars = [np.zeros(nb_bins, dtype=np.uint32)],
                                        concatenate_filter = concatenate_hist,
                                        context_manager = eomanager,
                                        filter_desc= "Compute histogram...")
    else:
        hist = eoexe.n_images_to_m_scalars(inputs = [input_dsm_key],
                                        filter_parameters = hist_params,
                                        image_filter = compute_hist,
                                        nb_output_scalars = 1,
                                        output_scalars = [np.zeros(nb_bins, dtype=np.uint32)],
                                        concatenate_filter = concatenate_hist,
                                        context_manager = eomanager,
                                        filter_desc= "Compute histogram...")

    # Compute the mean value of the histogram bin and determine the first bin greater than the mean (it is experimentaly the ground)
    robust_min_z, robust_max_z = compute_robust_z_interval(hist = hist[0],
                                             dsm_min_z = dsm_min,
                                             bin_width = bin_width,
                                             dsm_z_precision = dsm_z_precision)
    robust_max_z = dsm_max

    # Compute the uncertain mask and flush it to disk
    [noisy_mask] = eoexe.n_images_to_m_images_filter(inputs = [input_dsm_key],
                                                         image_filter = compute_uncertain_mask,
                                                         filter_parameters = {"min_z": robust_min_z, "max_z": robust_max_z},
                                                         generate_output_profiles = uncertain_profile,
                                                         context_manager = eomanager,
                                                         stable_margin = 0,
                                                         filter_desc= "Uncertain mask processing...")
    
    return {
        "noisy_mask": noisy_mask,
        "robust_min_z": robust_min_z,
        "robust_max_z": robust_max_z
    }

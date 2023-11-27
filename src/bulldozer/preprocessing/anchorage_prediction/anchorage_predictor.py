import bulldozer.eoscale.manager as eom
import bulldozer.preprocessing.anchors_predictor as pred

def run(filled_dsm_key: str,
        regular_mask_key: str,
        refined_min_z: float,
        refined_max_z: float,
        max_object_size: float,
        eomanager: eom.EOContextManager) -> str :
    

    anchors_mask_key = eomanager.create_image(profile = eomanager.get_profile(key = regular_mask_key))

    predictor = pred.PyAnchoragePredictor()
    predictor.predict(dsm = eomanager.get_array(key = filled_dsm_key),
                      regular_mask = eomanager.get_array(key = regular_mask_key),
                      anchors_mask = eomanager.get_array(key = anchors_mask_key),
                      refined_min_z = refined_min_z,
                      refined_max_z = refined_max_z,
                      nodata = eomanager.get_profile(key = filled_dsm_key)["nodata"],
                      max_object_size = max_object_size)
    
    return anchors_mask_key
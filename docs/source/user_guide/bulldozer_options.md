# Bulldozer options

This page provides a comprehensive reference of all available parameters in **Bulldozer**.  
These options allow users to control the behavior of the processing pipeline, adapt the tool to different types of input data, and fine-tune the DTM extraction results.

---

## Quick Navigation

### Options
- [generate_ndsm](#generate_ndsm)
- [max_object_size](#max_object_size)
- [ground_mask_path](#ground_mask_path)
- [activate_ground_anchors](#activate_ground_anchors)
- [nb_max_workers](#nb_max_workers)
- [developer_mode](#developer_mode)

### Expert Options
- [reg_filtering_iter](#reg_filtering_iter)
- [dsm_z_accuracy](#dsm_z_accuracy)
- [max_ground_slope](#max_ground_slope)
- [prevent_unhook_iter](#prevent_unhook_iter)
- [num_outer_iter](#num_outer_iter)
- [num_inner_iter](#num_inner_iter)
- [mp_context](#mp_context)
- [intermediate_write](#intermediate_write)


---

## Options

### generate_ndsm
**Description:** Generate the Digital Height Model (nDSM = DSM - DTM)

#### Example
- generate_ndsm = True  
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/tutorials/tutorial_LiDAR_result_DHM_Nice.png" alt="nDSM" width="400"/>
</div>

---

### max_object_size
**Description:** Maximum size of foreground objects (in meters)

#### Examples
- **Example 1:** max_object_size = 8
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/user_guide/bulldozer_options/dtm_max_obj_size_8.PNG" alt="max object size 8" width="400"/>
</div>

- **Example 2:** max_object_size = 16
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/user_guide/bulldozer_options/dtm_max_obj_size_16.PNG" alt="max object size 16" width="400"/>
</div>

- **Example 3:** max_object_size = 32
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/user_guide/bulldozer_options/dtm_max_obj_size_32.PNG" alt="max object size 32" width="400"/>
</div>

---

### ground_mask_path
**Description:** Path to the binary ground classification mask

#### Example
- Example of ground mask
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/user_guide/bulldozer_options/ground_mask.PNG" alt="ground mask" width="400"/>
</div>

- Dtm without ground mask
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/user_guide/bulldozer_options/dtm_without_ground_mask.PNG" alt="DTM without ground mask" width="400"/>
</div>

- Dtm with ground mask
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/user_guide/bulldozer_options/dtm_with_ground_mask.PNG" alt="DTM with ground mask" width="400"/>
</div>

---

### activate_ground_anchors
**Description:** Enable ground anchor detection

#### Examples
- **Example 1:** activate_ground_anchors = True
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/user_guide/bulldozer_options/dtm_ground_anchors.PNG" alt="DTM with ground anchors" width="400"/>
</div>

- **Example 2:** activate_ground_anchors = False
<div align="center">
<img src="https://raw.githubusercontent.com/CNES/bulldozer/master/docs/source/images/user_guide/bulldozer_options/dtm_max_obj_size_16.PNG" alt="DTM without ground anchors" width="400"/>
</div>


---

### nb_max_workers
**Description:** Maximum number of CPU cores to use

---

### developer_mode
**Description:** Keep intermediate results


---

## Expert options

!!! warning
    These parameters are intended for advanced users. It is recommended to keep the default values.

---

### reg_filtering_iter
**Description:** Number of regular mask filtering iterations

---

### dsm_z_accuracy
**Description:** Altimetric height accuracy of the input DSM

---

### max_ground_slope
**Description:** Maximum slope of the terrain

---

### prevent_unhook_iter
**Description:** Number of unhook iterations

---

### num_outer_iter
**Description:** Number of gravity iterations

---

### num_inner_iter
**Description:** Number of tension iterations

---

### mp_context
**Description:** Multiprocessing context

---

### intermediate_write
**Description:** Write intermediate results to disk

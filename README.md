# mymmdetection3d
My fork and modified version of OpenMMLab's mmdetection3d for general 3D object detection.

## Install the mymmdetection3d on SJSU HPC
The original mmdetection3d does not work on SJSU HPC, you need to use our forked version to install mymmdetection3d on SJSU HPC

## Dataset

### Waymo dataset conversion and generate infos.pkl
[class Waymo2KITTI](/tools/data_converter/waymo_converter.py) converts the Waymo dataset to Kitti format. [myconvert_waymo2kitti.py](/tools/myconvert_waymo2kitti.py) used [class Waymo2KITTI](/tools/data_converter/waymo_converter.py) to convert the waymo dataset and create the info (.pkl) files
* [def create_waymo_info_file](/tools/data_converter/kitti_converter.py) create info file of waymo dataset, it mainly calls [get_waymo_image_info](/tools/data_converter/kitti_data_utils.py) and [_calculate_num_points_in_gt](/tools/data_converter/kitti_data_utils.py) in kitti_data_utils.py
* [get_waymo_image_info](/tools/data_converter/kitti_data_utils.py) creates info dictionary for each idx, return waymo_infos_train is a list for idx, each idx element is a info dictionary data for all data
file save to waymo_infos_train.pkl
* [_calculate_num_points_in_gt](/tools/data_converter/kitti_data_utils.py) used information in the info dictionary and get velodyne points. It can box_np_ops.remove_outside_points and obtain 3D bounding box in the Velodyne coordinate via gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c). Finally, it gets annos['num_points_in_gt']

The detailed info dictionary generated in waymo_infos_train.pkl is
info['image']=image_info 
* image_info['image_path']
* image_info['image_shape']
* image_info['image_idx']: idx
info['point_cloud'] = pc_info
* pc_info['velodyne_path']
info['timestamp'] # get it from the last line of velodyne (not available in our conversion)
info['calib'] = calib_info (actual calibration data)
info['pose'] actual pose data
info['annos'] actual annotations
info['annos']['camera_id']
info['sweeps'] previous multiple velodyne path


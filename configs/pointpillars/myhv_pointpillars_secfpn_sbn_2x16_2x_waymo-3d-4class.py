_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_kitti.py',
    '../_base_/datasets/mywaymoD5-3d-4class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
] #'../_base_/models/hv_pointpillars_secfpn_waymo.py',

# data settings
#data = dict(train=dict(dataset=dict(load_interval=1))) # load one frame every ? frames
total_epochs =60 #80 #40
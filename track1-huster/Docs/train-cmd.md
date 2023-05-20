
这里记录了用于训练的命令


**挂载到后台训练**：

1、首先需要进入mmdetection文件夹：
    cd openMMLab/mmdetection
    
2、执行训练命令（挂到服务器后台, 8卡）：
    nohup bash tools/dist_train.sh ./config_track1/train/multi-task_dino_mask2former_internimage-xl_submit-trainV2.py 8 > tem.log 2>&1 &
    
    其中，./config_track1/train/multi-task_dino_mask2former_internimage-xl_submit-trainV2.py 是用于复现A榜精度的配置文件，精度应该比A榜更高。
    
**注意**：
1、训练日志及模型权重文件都默认保存在 openMMLab/work_dirs 下。
2、multi-task_dino_mask2former_internimage-xl_submit-trainV2.py 中一些标签路径和数据路径需要手动填入。
3、在A榜时我们同时使用了train和val的所有数据。
4、请确保训练时，预训练参数被正确加载。

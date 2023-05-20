
这里记录了用于测试的命令

**普通非TTA测试（一般用于TTA测试前和训练时精度对齐）**
1、首先需要进入mmdetection文件夹：
    cd openMMLab/mmdetection

2、执行测试命令（8卡）：
    bash tools dist_test.sh ./config_track1/train/multi-task_dino_mask2former_internimage-xl_submit-trainV2.py ../work_dirs/submit-trainV2/best.pth 8
    
    其中，./config_track1/train/multi-task_dino_mask2former_internimage-xl_submit-trainV2.py 是训练这个模型时使用的配置文件呢，
    ../work_dirs/submit-trainV2/best.pth 是测试使用的模型权重文件
    
    
**TTA测试（用于获得最终结果的TTA测试）**
1、首先需要进入mmdetection文件夹：
    cd openMMLab/mmdetection

2、执行TTA测试命令（8卡，测试时间较长，推荐挂到后台测试）：
    bash tools/dist_multitask_test.sh ./config_track1/test/multi-task_dino_mask2former_internimage-xl_submit-trainV2-TTA-test.py ../work_dirs/submit-trainV2/best.pth 8 --tta
    
    其中，dist_multitask_test.sh 是专门用于TTA测试的脚本，multi-task_dino_mask2former_internimage-xl_submit-trainV2-TTA-test.py 是专门用于TTA测试的配置文件
    ../work_dirs/submit-trainV2/best.pth 是TTA测试使用的模型权重。


**注意**：**使用非TTA和TTA都测试一遍后，选择单项任务的最优结果**，即：
final_det = max(no-tta-det, tta-det)
final_seg = max(no-tta-seg, tta-seg)
final_cls = max(no-tta-cls, tta-cls)



**使用 leaderboardA 的模型权重复现结果**
1、首先确保 openMMLab/params/iter-29480_submit.pth 模型权重已下载。
2、cd openMMLab/mmdetection
3、
bash tools/dist_multitask_test.sh config_track1/test/multi-task_dino_mask2former_internimage-xl_test-TTA_leaderboardA.py ../params/iter-29480_submit.pth 8 --tta
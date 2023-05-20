



openMMLab/mmdetection/config_track1 下都是用来训练和val 的 cfg 文件



test(用于普通测试和TTA测试的cfg文件)：

1、multi-task_dino_mask2former_internimage-xl_test-TTA_leaderboardA.py
用于复现A榜精度，TTA测试，需要和 openMMLab/mmdetection/tools/dist_multitask_test.sh 配合使用

2、multi-task_dino_mask2former_internimage-xl_test-no-TTA_leaderboardA.py
用于复现A榜精度，非TTA测试，需要和 openMMLab/mmdetection/tools/dist_test.sh 配合使用



train：
用于训练的cfg文件

1、multi-task_dino_mask2former_internimage-xl_submit-trainV2.py
提交用于复现A榜精度的cfg, 如果整体精度低于A榜, 请及时联系。理论精度应该优于A榜，A榜模型只训练了不到30000个iter，按照schedule需要训练 70000+ iter。
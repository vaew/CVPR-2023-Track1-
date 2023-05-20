

**用于从log中选择出最优的模型权重文件**

在 openMMLab/work_dirs 中会有路径类似于：
**work_dirs/multi-task_dino_mask2former_internimage-b_st18/20230517_032513/vis_data/20230517_032513.json**
的训练关键信息记录json文件。

修改 tools/find_best_iter.py 中的 rst_scalars_json_filename 变量，使之指向**上述json文件的绝对路径**

之后 python find_best_iter.py ，可以获得整体指标最好的模型权重文件。

**NOTE**: 已经在 find_best_iter.py 中填入了 log/20230517_032513 中的 json文件，作为示范。
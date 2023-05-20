
**CVPR2023 1st fondation model Track1 代码提交 队名：huster**

**队员**：
1、张泽伦，手机: 15927664956（微信同号，有问题烦请随时联系）
2、潘雪


**文件夹及文件介绍**：

Docs（一些说明文件）：
    1、set-env.md，训练及测试环境配置说明
    2、train-cmd.md，训练执行命令
    3、test-cmd.md，TTA测试及非TTA测试命令，最终结果选择两者中最优。
    4、tool_find-best-iter.md，使用工具分析训练过程关键变量获得整体指标最好的模型权重

openMMLab（训练代码，推理代码，及TTA推理代码）：
    mmengine
    mmdetection
    mmsegmentation
    mmpretrain
    work_dirs(用来保存训练参数以及log)
    params(用来存放A榜模型复现精度，和训练使用的预训练权重)


tools（一些结果分析文件）：
    find_best_iter.py , 用于解析训练log，获得整体指标最好的ep


data(训练时使用的标签文件，只有标签文件，没有图像数据，训练时train和val一起使用)：
    cls_train_and_val-amend.txt, 分类的训练标签
    cls_val.txt, 分类的验证标签
    det_train_val.json, 检测的训练标签
    det_val.json, 检测的验证标签
    NOTE: 分割的标签是图像，所以不在其中。

log(训练及测试log，都在同一个.log中)：
    20230517_032513, 训练过程中产生的训练和测试log, 仅仅作为示范，该log中检测mAP50到97.3, 分割mIoU到71.13, 分类ACC@top1到95.95, 都是非TTA的结果，**复现的效果应该比这个好很多, 检测应该进一步提升**
    log/20230517_032513/20230517_032513.log 可以看到整个训练和验证的表现。
    log/20230517_032513/vis_data/20230517_032513.json 记录了整个训练周期的关键数据。



**整体使用方法**：

从百度网盘链接：
下载：
链接: https://pan.baidu.com/s/1EALG8GjNvrb1SFJxcc_cwQ  密码: 4akr
1、internimage_xl_22kto1k_384_.pth, 模型训练的预训练权重。
2、iter-29480_submit.pth, A榜精度复现权重。
将上述两个文件放在 openMMLab/params 下

A榜模型精度测试：
1、具体方法见 Docs/test-cmd.md。


整体训练流程：
1、按照 Docs/set-env.md 中的要求配置代码需要的运行环境。
2、按照 Docs/train-cmd.md 中的命令，开始训练过程。
3、训练结束后，按照 Docs/tool_find-best-iter.md 找到整体指标最高的模型参数。
4、按照 Docs/test-cmd.md 中非TTA的命令复现3中选择的参数在训练中显示的精度，确实是这个参数。
5、按照 Docs/test-cmd.md 中的TTA的命令获得模型的最终精度。

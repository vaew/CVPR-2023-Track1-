
这里记录了训练的依赖环境，按照步骤安装即可配置好训练环境

**步骤**：

1、新建一个**python版本3.8.10**的conda虚拟环境，环境名可叫 **huster**。激活这个conda环境。
    conda create --name huster python=3.8.10
    conda activate huster

2、安装**版本为1.12.1+cu113的pytorch**，cuda版本推荐11.3，但是**pytorch版本一定要是1.12.1**。
    我们使用的安装命令：
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    
3、安装**mmengine和mmcv**。
    安装mim:                         pip install -U openmim
    用mim安装mmengine:    mim install mmengine
    用mim安装mmcv:            mim install "mmcv>=2.0.0"
    
    mim也可以使用清华源提高国内下载速度，如 mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    
4、直接**用 openMMLab/ mmengine 替换 conda环境huster 中的 mmengine**。
    mv openMMLab/mmengine  /path/to/your/anaconda3/envs/huster/lib/python3.8/site-packages/mmengine
    对mmengine做了一些多任务及多任务梯度协调的修改，所以需要用当前文件夹中的mmengine替换conda中的mmengine。

5、安装mmdetection
    cd openMMLab/mmdetection
    pip install -v -e .

6、安装mmsegmentation
    cd openMMLab/mmsegmentation
    pip install -v -e .

7、安装mmpretrain
    cd openMMLab / mmpretrain
    mim install -e .

8、安装一些其他的依赖库：
    pip install albumentations
    pip install imgaug
    pip install imagecorruptions


**测试**：
1、启动conda环境huster下的python
2、import mmcv; import mmengine; import mmdet; import mmseg; import mmpretrain
3、mmcv.__version__; mmengine.__version__; mmdet.__version__; mmseg.__version__; mmpretrain.__version__
期望输出：
'2.0.0'
'0.7.2'
'3.0.0'
'1.0.0'
'1.0.0rc7'

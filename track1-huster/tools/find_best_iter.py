#coding=utf-8



'''找到结果最好的iter
'''

import json
import os


# 训练结束后，保存每个iter和val结果的json文件
rst_scalars_json_filename = 'log/20230517_032513/vis_data/20230517_032513.json'

indicators = ['det_coco/bbox_mAP_50', 'seg_mIoU', 'cls_accuracy/top1']


def _is_validation_line(line: str):
    '''判断json_filename中这一行是不是存储validation结果的line
    '''
    return line.startswith('{"det_coco')


def load_multi_dict_json(json_filename):
    '''加载mmlab生成的每次测试的关键信息
    '''
    assert os.path.isfile(json_filename)
    validation_rst = []
    with open(json_filename) as tem:
        lines = tem.readlines()
    for line in lines:
        if _is_validation_line(line):
            validation_rst.append(json.loads(line.rstrip('\n')))
    return validation_rst

def main():
    validation_rst = load_multi_dict_json(rst_scalars_json_filename)
    print(f'total {len(validation_rst)} validation results.')
    best_ep = -1
    best_overall_metric = -1
    best_ep_metric = {ind: -1 for ind in indicators}
    for ep_idx, rst in enumerate(validation_rst):
        ep_metric = []
        for ind in indicators:
            ind_rst = rst[ind]
            if ind_rst > 1: ind_rst *= 0.01 # 如果指标带百分号，则去掉百分号
            ep_metric.append(ind_rst)

        ep_overall = sum(ep_metric) / len(ep_metric) # 这个ep的整体metric
        
        if ep_overall > best_overall_metric:
            best_ep = ep_idx
            best_overall_metric = ep_overall
            for ind in indicators:
                ind_rst = rst[ind]
                if ind_rst > 1: ind_rst *= 0.01
                best_ep_metric[ind] = ind_rst
    
    print('best epoch index (index start from 1):', best_ep + 1)
    print('best overall metric:', best_overall_metric)
    print('best single task metric:', best_ep_metric)




if __name__ == '__main__':
    main()
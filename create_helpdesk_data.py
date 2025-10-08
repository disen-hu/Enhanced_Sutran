#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpdesk数据集构建脚本
基于create_road_traffic_data.py修改，适配helpdesk.csv数据集
"""

import pandas as pd
import numpy as np
import os
import sys
import torch
from datetime import datetime

# 添加Preprocessing目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'Preprocessing'))
from from_log_to_tensors import log_to_tensors

def preprocess_helpdesk(df):
    """
    预处理helpdesk数据集
    
    Args:
        df: 原始数据框
    
    Returns:
        处理后的数据框
    """
    print("开始预处理helpdesk数据...")
    
    # 复制数据框避免修改原始数据
    df = df.copy()
    
    # 重命名列以符合标准格式
    column_mapping = {
        'Case ID': 'case:concept:name',
        'Activity': 'concept:name', 
        'Resource': 'org:resource',
        'Complete Timestamp': 'time:timestamp'
    }
    
    df = df.rename(columns=column_mapping)
    
    # 转换时间戳
    try:
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        print(f"成功转换时间戳列")
    except Exception as e:
        print(f"转换时间戳时出错: {e}")
        raise
    
    # 按案例ID和时间戳排序
    df = df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    
    # 处理数值列 - 这些列可能包含数值信息
    potential_numeric_cols = ['Variant index']
    
    for col in potential_numeric_cols:
        if col in df.columns:
            # 替换常见的缺失值表示
            df[col] = df[col].replace(['NIL', 'NULL', 'N/A', 'nan', 'NaN', '', ' '], np.nan)
            
            # 尝试转换为数值类型
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 用0填充NaN值
                df[col] = df[col].fillna(0)
                print(f"成功处理数值列: {col}")
            except Exception as e:
                print(f"处理数值列 {col} 时出错: {e}")
    
    # 处理分类列
    categorical_cols = ['concept:name', 'org:resource', 'Variant', 'seriousness', 
                       'customer', 'product', 'responsible_section', 'seriousness_2',
                       'service_level', 'service_type', 'support_section', 'workgroup']
    
    for col in categorical_cols:
        if col in df.columns:
            # 转换为字符串并填充缺失值
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(['nan', 'NaN', 'None', ''], 'Missing')
            print(f"成功处理分类列: {col}")
    
    print(f"预处理完成。数据形状: {df.shape}")
    return df

def construct_helpdesk_datasets(csv_path):
    """
    构建helpdesk数据集的训练、验证和测试张量
    
    Args:
        csv_path: CSV文件路径
    """
    print(f"开始构建helpdesk数据集: {csv_path}")
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件，数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    # 预处理数据
    df = preprocess_helpdesk(df)
    
    # 定义列映射
    case_id = 'case:concept:name'
    act_label = 'concept:name'
    timestamp = 'time:timestamp'
    
    # 动态识别特征列
    all_columns = set(df.columns)
    reserved_columns = {case_id, act_label, timestamp}
    feature_columns = all_columns - reserved_columns
    
    # 分类特征和数值特征
    categorical_casefeatures = []
    num_casefts = []
    categorical_eventfeatures = []
    numeric_eventfeatures = []
    
    # 根据数据类型自动分类特征
    for col in feature_columns:
        if df[col].dtype in ['object', 'category']:
            categorical_eventfeatures.append(col)
        else:
            numeric_eventfeatures.append(col)
    
    print(f"\n=== 特征分类 ===")
    print(f"分类事件特征: {categorical_eventfeatures}")
    print(f"数值事件特征: {numeric_eventfeatures}")
    
    # 时间相关设置
    start_date = None
    end_date = None
    max_days = 365  # 设置最大案例持续时间为365天

    # 超参数设置
    window_size = 17  # 可能需要调整的超参数
    log_name = 'Helpdesk_data'  # 输出目录名
    start_before_date = None
    test_len_share = 0.25
    val_len_share = 0.2
    mode = 'workaround'  # 如BPIC19脚本中使用的
    outcome = None  # 如果有预测任务的结果列，请定义

    # 打印配置信息
    print(f"\n=== 数据集构建配置 ===")
    print(f"使用case_id: {case_id}")
    print(f"使用timestamp: {timestamp}")
    print(f"使用act_label: {act_label}")
    print(f"案例级分类特征: {categorical_casefeatures}")
    print(f"案例级数值特征: {num_casefts}")
    print(f"事件级分类特征: {categorical_eventfeatures}")
    print(f"事件级数值特征: {numeric_eventfeatures}")
    print(f"窗口大小: {window_size}")
    print(f"测试集比例: {test_len_share}")
    print(f"验证集比例: {val_len_share}")

    # 数据统计
    print(f"\n=== 数据统计 ===")
    print(f"总事件数: {len(df)}")
    print(f"总案例数: {df[case_id].nunique()}")
    print(f"活动类型数: {df[act_label].nunique()}")
    print(f"\n活动类型分布:")
    activity_counts = df[act_label].value_counts()
    for activity, count in activity_counts.head(10).items():
        print(f"  {activity}: {count}")

    # 调用log_to_tensors函数
    print(f"\n开始转换为张量...")
    try:
        result = log_to_tensors(df,
                                log_name=log_name,
                                start_date=start_date,
                                start_before_date=start_before_date,
                                end_date=end_date,
                                max_days=max_days,
                                test_len_share=test_len_share,
                                val_len_share=val_len_share,
                                window_size=window_size,
                                mode=mode,
                                case_id=case_id,
                                act_label=act_label,
                                timestamp=timestamp,
                                cat_casefts=categorical_casefeatures,
                                num_casefts=num_casefts,
                                cat_eventfts=categorical_eventfeatures,
                                num_eventfts=numeric_eventfeatures,
                                outcome=outcome)

        train_data, val_data, test_data = result
        print("张量转换成功!")

    except Exception as e:
        print(f"张量转换时出错: {e}")
        print("请检查数据格式和特征配置")
        raise

    # 创建输出目录并保存数据
    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    # 保存训练数据
    train_tensors_path = os.path.join(output_directory, 'train_tensordataset.pt')
    torch.save(train_data, train_tensors_path)
    print(f"训练数据已保存到: {train_tensors_path}")

    # 保存验证数据
    val_tensors_path = os.path.join(output_directory, 'val_tensordataset.pt')
    torch.save(val_data, val_tensors_path)
    print(f"验证数据已保存到: {val_tensors_path}")

    # 保存测试数据
    test_tensors_path = os.path.join(output_directory, 'test_tensordataset.pt')
    torch.save(test_data, test_tensors_path)
    print(f"测试数据已保存到: {test_tensors_path}")

    print(f"\n=== 数据集构建完成 ===")
    print(f"输出目录: {output_directory}")
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")

if __name__ == "__main__":
    # 设置CSV文件路径
    helpdesk_csv_file_path = r'c:\Users\disen\Desktop\root\autodl-tmp\sutran\SuffixTransformerNetwork-main\helpdesk.csv'
    
    # 检查文件是否存在
    if not os.path.exists(helpdesk_csv_file_path):
        print(f"错误: CSV文件不存在: {helpdesk_csv_file_path}")
        print("请确认文件路径是否正确")
        sys.exit(1)
    
    # 构建数据集
    construct_helpdesk_datasets(helpdesk_csv_file_path)
    
    print("\n=== 使用说明 ===")
    print("1. 数据集已成功构建并保存为PyTorch张量")
    print("2. 可以使用以下代码加载数据:")
    print("   train_data = torch.load('Helpdesk_data/train_tensordataset.pt')")
    print("   val_data = torch.load('Helpdesk_data/val_tensordataset.pt')")
    print("   test_data = torch.load('Helpdesk_data/test_tensordataset.pt')")
    print("3. 要确定tss_index，请检查生成的num_cols_dict文件")
    print("4. 可以参考TRAIN_EVAL_*.py脚本来训练模型")
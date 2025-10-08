#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Traffic Fine Management Process Dataset Constructor
为Road Traffic Fine Management Process数据创建训练用的数据集
参考create_sepsis_data.py的处理流程
"""

import pandas as pd
import numpy as np
from Preprocessing.from_log_to_tensors import log_to_tensors
import os
import torch

def preprocess_road_traffic(log):
    """Preprocess the Road Traffic Fine Management Process event log.

    Parameters
    ----------
    log : pandas.DataFrame
        Event log.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """
    print("开始预处理Road Traffic数据...")
    
    # 转换时间戳格式
    # Road Traffic数据可能使用不同的时间格式，需要灵活处理
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], errors='coerce', utc=True)

    # 检查时间戳转换后的NaT值
    nat_count = log['time:timestamp'].isnull().sum()
    if nat_count > 0:
        print(f"警告: 在时间戳转换后发现 {nat_count} 个NaT值")
        print("这可能表明某些时间戳数据格式存在问题")
        print("具有NaT时间戳的行在基于时间的操作中会出现问题")

    # 检查时间戳列是否为datetime类型
    if pd.api.types.is_datetime64_any_dtype(log['time:timestamp']):
        pass  # 已经通过pd.to_datetime的utc=True转换为UTC
    elif not log['time:timestamp'].isnull().all():
        print("警告: 'time:timestamp'列在初始转换后不是统一的datetime类型，但不是全部为NaT")
    else:
        print("错误: 'time:timestamp'列无法转换为datetime格式或完全由NaT值组成")
        print("进一步的基于时间的处理可能会失败，请检查CSV中的'time:timestamp'列")

    # 按case_id和时间戳排序
    if 'time:timestamp' in log.columns and pd.api.types.is_datetime64_any_dtype(log['time:timestamp']) and not log['time:timestamp'].isnull().all():
        log.sort_values(by=['case:concept:name', 'time:timestamp'], inplace=True)
    else:
        print("警告: 由于时间戳列存在问题，跳过按'time:timestamp'排序，仅按'case:concept:name'排序")
        log.sort_values(by=['case:concept:name'], inplace=True)

    # 处理数值列中的潜在NaN值
    # Road Traffic数据可能包含不同的数值特征，需要根据实际数据调整
    numeric_cols_to_check = []
    
    # 检查常见的数值列
    potential_numeric_cols = ['Amount', 'Points', 'Fine', 'Speed', 'Limit', 'Expense', 
                             'amount', 'article', 'dismissal', 'expense', 'lastSent', 
                             'matricola', 'paymentAmount', 'points', 'totalPaymentAmount']
    for col in potential_numeric_cols:
        if col in log.columns:
            numeric_cols_to_check.append(col)
    
    # 处理数值列
    for col in numeric_cols_to_check:
    # 替换特殊字符串值
        log[col] = log[col].replace(['NIL', 'NULL', 'N/A', 'nan', 'NaN', '', ' '], np.nan)
        log[col] = pd.to_numeric(log[col], errors='coerce').fillna(0)
        print(f"处理数值列: {col}")

    # 确保分类特征为字符串类型
    categorical_cols_to_fill_as_str = [
        'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition', 'org:group',
        'concept:name', 'case:concept:name'
    ]
    
    # 检查并处理分类列
    for col in categorical_cols_to_fill_as_str:
        if col in log.columns:
            log[col] = log[col].astype(str).fillna('Missing')
        else:
            print(f"警告: 分类列'{col}'在日志中未找到")

    # 处理其他可能的分类列
    for col in log.columns:
        if col not in categorical_cols_to_fill_as_str and col not in numeric_cols_to_check and col != 'time:timestamp':
            if log[col].dtype == 'object':
                log[col] = log[col].astype(str).fillna('Missing')
                print(f"处理额外分类列: {col}")

    print(f"预处理完成，共有 {len(log)} 个事件")
    return log

def construct_road_traffic_datasets(csv_path):
    """构建Road Traffic Fine Management Process数据集
    
    Parameters
    ----------
    csv_path : str
        CSV文件路径
    """
    print(f"开始构建Road Traffic数据集，CSV路径: {csv_path}")
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {list(df.columns)}")
    
    # 预处理数据
    df = preprocess_road_traffic(df)

    # 核心列名 - 这些应该与您的CSV匹配
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name'  # 活动标签

    # 检查必需列是否存在
    required_cols = [case_id, timestamp, act_label]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"必需列'{col}'在CSV中未找到")

    # 动态识别特征列
    print("\n动态识别特征列...")
    
    # 案例级别的分类特征
    categorical_casefeatures = []
    # Road Traffic数据中可能的案例级别特征
    potential_case_features = ['case:Expense', 'case:Amount', 'case:Points']
    for col in potential_case_features:
        if col in df.columns:
            # 检查是否为分类特征（唯一值较少）
            unique_vals = df[col].nunique()
            if unique_vals < 50:  # 阈值可调整
                categorical_casefeatures.append(col)
                print(f"识别案例级分类特征: {col} (唯一值: {unique_vals})")
    
    # 案例级别的数值特征
    num_casefts = []
    potential_case_numeric = ['case:Expense', 'case:Amount', 'case:Points']
    for col in potential_case_numeric:
        if col in df.columns and col not in categorical_casefeatures:
            try:
                pd.to_numeric(df[col], errors='raise')
                num_casefts.append(col)
                print(f"识别案例级数值特征: {col}")
            except:
                pass

    # 事件级别的分类特征
    categorical_eventfeatures = []
    event_cat_cols = ['org:resource', 'Action', 'EventOrigin', 'lifecycle:transition', 'org:group']
    for col in event_cat_cols:
        if col in df.columns:
            categorical_eventfeatures.append(col)
            print(f"识别事件级分类特征: {col}")

    # 事件级别的数值特征
    numeric_eventfeatures = []
    # 检查所有数值列
    for col in df.columns:
        if col not in [case_id, timestamp, act_label] and col not in categorical_eventfeatures and col not in categorical_casefeatures and col not in num_casefts:
            try:
                # 尝试转换为数值
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isnull().all():  # 如果不是全部为NaN
                    numeric_eventfeatures.append(col)
                    print(f"识别事件级数值特征: {col}")
            except:
                pass
    
    # 从数据动态确定日期范围和max_days
    if not df[timestamp].isnull().all():
        actual_min_date = df[timestamp].min()
        actual_max_date = df[timestamp].max()
        
        if pd.notna(actual_min_date) and pd.notna(actual_max_date):
            start_date = actual_min_date.strftime('%Y-%m-%d')
            end_date = actual_max_date.strftime('%Y-%m-%d')
            max_days = (actual_max_date - actual_min_date).days
            print(f"动态确定开始日期: {start_date}")
            print(f"动态确定结束日期: {end_date}")
            print(f"动态确定最大天数: {max_days}")
        else:
            print("警告: 无法从数据确定有效的最小/最大日期，日期参数使用None")
            start_date = None
            end_date = None
            max_days = None
    else:
        print("警告: 时间戳列全部为NaT或为空，日期参数使用None")
        start_date = None
        end_date = None
        max_days = None

    # 超参数设置
    window_size = 17  # 可能需要调整的超参数
    log_name = 'Road_Traffic_data'  # 输出目录名
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

def main():
    """主函数"""
    # Road Traffic CSV文件路径
    road_traffic_csv_file_path = r'c:\Users\disen\Desktop\root\autodl-tmp\sutran\SuffixTransformerNetwork-main\road_traffic_data.csv'
    
    # 检查Preprocessing模块和log_to_tensors是否可访问
    try:
        from Preprocessing.from_log_to_tensors import log_to_tensors
        print("成功导入log_to_tensors")
    except ImportError as e:
        print(f"导入log_to_tensors时出错: {e}")
        print("请确保'Preprocessing/from_log_to_tensors.py'在正确路径且所有依赖项已安装")
        exit()

    # 检查CSV文件是否存在
    if not os.path.exists(road_traffic_csv_file_path):
        print(f"错误: CSV文件不存在: {road_traffic_csv_file_path}")
        print("请先使用convert_road_traffic_xes_to_csv.py转换XES文件为CSV")
        print("或者修改road_traffic_csv_file_path变量指向正确的CSV文件路径")
        exit()

    # 构建数据集
    try:
        construct_road_traffic_datasets(road_traffic_csv_file_path)
        print("Road Traffic数据集构建完成!")
    except Exception as e:
        print(f"数据集构建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
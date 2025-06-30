import json
import pandas as pd
import os
import glob
from pathlib import Path


def process_json_files_to_csv():
    """
    处理多个文件夹中的JSON文件，将它们合并为一个大的DataFrame并保存为CSV
    """

    # 定义文件夹路径
    json_folders = [
        'data/LALSemiCoherentF0F1F2_aggressive_memory_mac',
        'LAL_example_data/LALSemiCoherentF0F1F2_aggressive_memory_codespace',
        'LAL_example_data/LALSemiCoherentF0F1F2_aggressive_memory'
    ]

    # 存储所有处理后的DataFrame
    all_dataframes = []

    for folder in json_folders:
        print(f"Processing folder: {folder}")

        # 检查文件夹是否存在
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist, skipping...")
            continue

        # 查找所有匹配的JSON文件
        pattern = os.path.join(folder, "config_mismatch_run_*.json")
        json_files = glob.glob(pattern)

        print(f"Found {len(json_files)} JSON files in {folder}")

        for json_file in json_files:
            try:
                # 读取JSON文件
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # 提取配置参数
                config = data.get('config', {})
                mf = config.get('mf')
                mf1 = config.get('mf1')
                mf2 = config.get('mf2')
                gamma1 = config.get('gamma1')
                gamma2 = config.get('gamma2')
                T_coh = config.get('T_coh')

                # 提取mismatch数据
                mismatch_list = data.get('mismatch_list', [])

                # 创建DataFrame：每行包含配置参数和一个mismatch值
                rows = []
                for i, mismatch in enumerate(mismatch_list):
                    row = {
                        'file_path': json_file,
                        'folder': folder,
                        'run_number': os.path.basename(json_file).replace('config_mismatch_run_', '').replace('.json', ''),
                        'mismatch_index': i,
                        'mf': mf,
                        'mf1': mf1,
                        'mf2': mf2,
                        'gamma1': gamma1,
                        'gamma2': gamma2,
                        'T_coh': T_coh,
                        'mismatch': mismatch
                    }
                    rows.append(row)

                # 创建当前文件的DataFrame
                df_current = pd.DataFrame(rows)
                all_dataframes.append(df_current)

                print(
                    f"Processed {json_file}: {len(mismatch_list)} mismatch values")

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue

    # 合并所有DataFrame
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # 显示统计信息
        print(f"\nTotal rows in combined DataFrame: {len(combined_df)}")
        print(f"Total JSON files processed: {len(all_dataframes)}")
        print(f"Columns: {list(combined_df.columns)}")

        # 显示数据预览
        print("\nData preview:")
        print(combined_df.head())

        print("\nData info:")
        print(combined_df.info())

        # 保存为CSV
        output_filename = 'data/combined_mismatch_data.csv'
        combined_df.to_csv(output_filename, index=False)
        print(f"\nData saved to: {output_filename}")

        return combined_df
    else:
        print("No data was processed successfully.")
        return None


def analyze_data(df):
    """
    对合并后的数据进行简单分析
    """
    if df is None:
        return

    print("\n=== Data Analysis ===")

    # 基本统计
    print("Mismatch statistics:")
    print(df['mismatch'].describe())

    # 按文件夹分组统计
    print("\nStatistics by folder:")
    folder_stats = df.groupby('folder')['mismatch'].agg(
        ['count', 'mean', 'std', 'min', 'max'])
    print(folder_stats)

    # 按参数组合统计
    print("\nUnique parameter combinations:")
    param_cols = ['mf', 'mf1', 'mf2', 'gamma1', 'gamma2', 'T_coh']
    unique_params = df[param_cols].drop_duplicates()
    print(f"Number of unique parameter combinations: {len(unique_params)}")
    print(unique_params)


    x_data = df[['mf', 'mf1', 'mf2', 'gamma1', 'gamma2', 'T_coh']]
    y_data = df['mismatch']
    
    x_data.to_csv('data/x_data2.csv', index=False)
    y_data.to_csv('data/y_data2.csv', index=False)

# 主函数
if __name__ == "__main__":
    # 处理JSON文件
    combined_data = process_json_files_to_csv()

    # 分析数据
    analyze_data(combined_data)

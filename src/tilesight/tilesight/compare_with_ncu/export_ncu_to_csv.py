import os
import subprocess
import pandas as pd

# 定义输出CSV的函数
def export_ncu_to_csv(ncu_report_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_csv_file = os.path.join(output_folder, f"{os.path.splitext(ncu_report_file)[0]}.csv")

    # 构建ncu命令，将.ncu-rep文件转换为CSV
    ncu_command = f"ncu --import {ncu_report_file} --csv --page raw > {output_csv_file}"
    
    # 执行命令
    subprocess.run(ncu_command, shell=True, check=True)
    return output_csv_file

if __name__ == "__main__":
    input_ncu_report = "profile.ncu-rep"
    
    
    output_directory = "."
    exported_csv = export_ncu_to_csv(input_ncu_report, output_directory)
    print(f"CSV file exported to: {exported_csv}")

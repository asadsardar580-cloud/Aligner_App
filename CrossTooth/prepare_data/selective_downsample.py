import os
import subprocess
import shutil
import pymeshlab as ml


EXE_PATH = "selective_downsample.exe"
FILE_ROOT = "ply_file_cell_color_manifold"

target_path = "selective_downsample_result"

processed_parts = os.listdir(target_path)

ERROR_CASE = []

for file in os.listdir(FILE_ROOT):
    file_path = os.path.join(FILE_ROOT, file)
    print(file_path)
    base_name = os.path.basename(file)[:-4]

    if base_name in processed_parts:
        continue

    folder = os.path.join(target_path, base_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    # 执行一个 exe 文件并传递参数
    curvature_file_path = os.path.join(folder, base_name + "_curvature.ply")
    selective_downsample_file_path = os.path.join(folder, base_name + "_selective_downsample.ply")
    print(curvature_file_path)
    print(selective_downsample_file_path)

    tmp_downsample_file_path = os.path.join(folder, "downsample")

    # 迭代次数
    round = "2"
    try:
        subprocess.run([EXE_PATH, file_path, curvature_file_path, tmp_downsample_file_path, round], shell=False)
    except Exception as e:
        print(e)
        ERROR_CASE.append(file_path)

    # 把结果移动到目标文件夹，清除中间结果
    for i in range(int(round)):
        inter_file = tmp_downsample_file_path + f"_round{i + 1}.ply"
        if os.path.exists(inter_file):
            if i + 1 == int(round):
                shutil.move(inter_file, selective_downsample_file_path)
            else:
                os.remove(inter_file)

    if os.path.exists(curvature_file_path):
        os.remove(curvature_file_path)

print(ERROR_CASE)
print("end")
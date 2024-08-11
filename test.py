# import tarfile
# #
# # tar 파일의 경로와 압축을 풀 폴더의 경로를 지정합니다.
file_name = '01_real_sen_video'
# tar_file_path = 'D://data//train//REAL//' + file_name + '.tar'  # 풀고자 하는 tar 파일 경로
# extract_path = 'D://data//train//REAL//'  # 압축을 풀 폴더 경로
#
# # tar 파일 열기
# with tarfile.open(tar_file_path, 'r') as tar:
#     tar.extractall(path=extract_path)
#
# print("압축 해제가 완료되었습니다!")


import os

# 파일 경로와 파일 이름 설정
directory = r'D://data//train//REAL//004.수어영상//1.Training//원천데이터//REAL//SEN' # SEN, WORD 없어야될수도
file_prefix = file_name + '.zip.part' # 뒤에 번호 x
output_file = os.path.join(directory, file_name + '.zip')

# # 결합할 파일 생성
# with open(output_file, 'wb') as outfile:
#     part_num = 0
#     while True:
#         part_file = os.path.join(directory, f'{file_prefix}{part_num}')
#         if not os.path.exists(part_file):
#             break
#
#         with open(part_file, 'rb') as infile:
#             outfile.write(infile.read())
#
#         part_num += 1
#
# print(f'{output_file} 파일로 결합이 완료되었습니다.')

# import zipfile
#
# # 결합된 파일의 경로 설정
# zip_file = os.path.join(directory, file_name + '.zip')
# print(zip_file)
# # 압축 풀기
# with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#     extract_path = os.path.join(directory, 'extracted_files')
#     zip_ref.extractall(extract_path)
#
# print(f'{zip_file} 압축 해제가 완료되었습니다.')

# import os
# import glob
#
# # 대상 디렉토리 경로
# directory_path = r"D:\data\train\REAL\004.수어영상\1.Training\원천데이터\REAL\SEN\01"
#
# # 모든 mp4 파일 찾기
# mp4_files = glob.glob(os.path.join(directory_path, "*.mp4"))
#
# # 'F'로 끝나지 않는 파일 삭제
# for file_path in mp4_files:
#     # 파일명만 추출
#     file_name = os.path.basename(file_path)
#
#     # 'F'로 끝나는지 확인
#     if not file_name.endswith("_F.mp4"):
#         print(f"Deleting file: {file_name}")
#         os.remove(file_path)  # 파일 삭제
#
# print("Cleanup completed.")

import cv2
import os
import glob

# 대상 디렉토리 경로
directory_path = r"D:\data\train\REAL\sign_mov\1.Training\video\REAL\SEN\01"

# 모든 mp4 파일 찾기
mp4_files = glob.glob(os.path.join(directory_path, "*.mp4"))

# 각 mp4 파일에 대해 작업 수행
for mp4_file in mp4_files:
    # 파일명과 확장자 분리
    file_name = os.path.basename(mp4_file)
    file_base_name = os.path.splitext(file_name)[0]

    # 파일명과 동일한 이름의 폴더 생성
    output_folder = os.path.join(directory_path, file_base_name)
    os.makedirs(output_folder, exist_ok=True)

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(mp4_file)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {mp4_file}")
        continue

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Completed processing {file_name} into folder {output_folder}")
            break
        frame_count += 1

        # 프레임을 이미지 파일로 저장
        frame_filename = f"{file_base_name}_{frame_count:03d}.jpg"
        success = cv2.imwrite(os.path.join(output_folder, frame_filename), frame)
        if not success:
            print(f"Error: Failed to write image {os.path.join(output_folder, frame_filename)}")

    cap.release()

print("All files have been processed.")



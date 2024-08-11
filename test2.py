import os
import json
import pandas as pd

# 경로 설정
video_folder_path = r"D:\data\train\REAL\sign_mov\1.Training\video\REAL\SEN\01"
json_folder_path = r"D:\data\train\004.수어영상\1.Training\라벨링데이터\REAL\SEN\extracted_files\morpheme\01"

# 결과를 저장할 리스트 초기화
data = []

# 모든 폴더 탐색
for folder_name in os.listdir(video_folder_path):
    folder_path = os.path.join(video_folder_path, folder_name)

    # 폴더인 경우
    if os.path.isdir(folder_path):
        # JSON 파일 경로 설정
        json_filename = f"{folder_name}_morpheme.json"
        json_filepath = os.path.join(json_folder_path, json_filename)

        # JSON 파일이 존재하는지 확인
        if os.path.exists(json_filepath):
            with open(json_filepath, 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)
                # "name" 필드 추출 (모든 attributes의 name 필드를 연결)
                names = [attr["name"] for entry in json_data["data"] for attr in entry["attributes"]]
                # 모든 "name"을 띄어쓰기로 연결
                kor_text = " ".join(names)
                # 데이터 추가
                data.append({"Filename": folder_name + ".mp4", "Kor": kor_text})
        else:
            print(f"JSON file not found: {json_filepath}")

# DataFrame으로 변환
df = pd.DataFrame(data)

# CSV 파일로 저장
output_csv_path = r"D:\data\output.csv"
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"CSV file saved to: {output_csv_path}")

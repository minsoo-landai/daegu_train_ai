#데이터 확인용 코드
import pandas as pd
import glob

csv_files = glob.glob("/home/minsoo0807/deagu_manufacture_ai/data/train_data/vib/*.csv")
sample_file = csv_files[0]

df = pd.read_csv(sample_file)
print(f"파일: {sample_file}")
print(f"컬럼: {df.columns.tolist()}")
print(f"행 수: {len(df)}")
print(f"첫 5행:")
print(df.head())
import pandas as pd
import csv

output = 'walkloradyna2.csv'

# CSVファイルの読み込み
df1 = pd.read_csv('trainseed0.csv')
df2 = pd.read_csv('trainseed1.csv')
df3 = pd.read_csv('trainseed2.csv')
# 横に連結
df_merged = pd.concat([df1, df2, df3], axis=1)

# 結果を新しいCSVファイルとして保存
df_merged.to_csv(output, index=False)

averages = []
"""
# まずは読み込みと計算だけを行う
with open(output, 'r', encoding='utf-8') as f_in:
    reader = csv.reader(f_in)

    for row in reader:
        numbers = []
        for val in row:
            try:
                numbers.append(float(val))
            except ValueError:
                pass
        
        if numbers:
            row_avg = sum(numbers) / len(numbers)
            averages.append(f"{row_avg:.2f}")
        else:
            averages.append("")

# 計算した平均値のリストを、新しいCSVに「1行だけ」書き込む
with open(output, 'w', encoding='utf-8', newline='') as f_out:
    writer = csv.writer(f_out)
    # writerowにリストを渡すと、そのリストが横1行として出力されます
    writer.writerow(averages)
    """
print("CSVファイルが横に連結されました。")

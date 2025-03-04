# 5×10のテキストファイルをパースしてnp.saveするだけのスクリプト
# Third Party Library
import numpy as np

# numpyに値を放り込む

calc_data = np.zeros((10, 5))
base_name = "dualcast_dblp"
f = open(base_name + ".log", "r")

datalist = f.readlines()
for i, data in enumerate(datalist):
    calc_data[int(i / 5)][int(i % 5)] = data

print(calc_data)

np.save(base_name, calc_data)

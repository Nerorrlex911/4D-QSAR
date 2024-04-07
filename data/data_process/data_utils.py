import pandas as pd
import logging

def appendDataLine(data:pd.DataFrame, dataLine:dict):
    """
    将dataLine的键和值作为两列添加到data中
    """

    # 创建一个新的DataFrame，包含dataLine的键和值
    new_data = pd.DataFrame(list(dataLine.items()), columns=['desc_signature', 'desc_amount'])

    # 使用pandas.concat()函数将data和new_data连接起来
    data = pd.concat([data, new_data], ignore_index=True)

    return data

def divide_list(l, n):
    """
    将列表均分为n份
    """
    # 计算每份的长度
    length = len(l)
    size = length // n
    if length % n:
        size += 1

    # 使用切片操作将列表均分为n份
    for i in range(0, length, size):
        yield l[i:i+size]
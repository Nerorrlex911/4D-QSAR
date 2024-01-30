import pandas as pd

def appendDataLine(data:pd.DataFrame, dataLine:dict):
    """
    将dataLine添加到data中
    """
    data[len(data)] = dataLine
    return data
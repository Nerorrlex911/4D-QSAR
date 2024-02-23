if __name__ == "__main__":
    # 导入pandas库
    import pandas as pd

    # 创建两个数据框
    testDataFrame = pd.DataFrame({
        'desc_signature': ['a', 'b', 'c', 'd', 'e'],
        'desc_amount': ['1', '2', '3', '4', '5']
    })
    testDataFrame['desc_amount'] = testDataFrame['desc_amount'].astype(int)
    testDataFrame2 = pd.DataFrame({
        'desc_signature': ['f', 'g', 'c', 'd', 'e'],
        'desc_amount': ['1', '2', '3', '4', '5']
    })
    testDataFrame2['desc_amount'] = testDataFrame2['desc_amount'].astype(int)

    # 将两个数据框连接起来
    combined = pd.concat([testDataFrame, testDataFrame2])

    

    # 对desc_signature相同的行进行求和
    result = combined.groupby('desc_signature', as_index=False).sum()

    print(result)
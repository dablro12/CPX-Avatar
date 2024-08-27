import numpy as np 

def remove_outliers(chunk_sizes):
    # 1사분위수(Q1)와 3사분위수(Q3) 계산
    Q1 = np.percentile(chunk_sizes, 25)
    Q3 = np.percentile(chunk_sizes, 75)
    
    # IQR 계산
    IQR = Q3 - Q1
    
    # 이상치 임계값 설정
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 이상치가 아닌 값들만 필터링
    filtered_chunk_sizes = [size for size in chunk_sizes if lower_bound <= size <= upper_bound]
    
    return filtered_chunk_sizes, lower_bound, upper_bound

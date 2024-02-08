def preprosessing(df):
    # inquiry_type feature 전처리 
    # customer_position feature 전처리 
    df['inquiry_type'] = df['inquiry_type'].str.lower()
    df['customer_position'] = df['customer_position'].str.lower()
    
    # 특수문자를 대체 문자열 지정
    replacement = {'/': ' ', '-':' ', '_':' '}
    # replace() 함수를 사용하여 특수 문자 대체
    df['inquiry_type'].replace(replacement, regex=True, inplace=True)
    df['customer_position'].replace(replacement, regex=True, inplace=True)
    
    # value count의 값이 1개인 feature name extraction (해당 방식 토의) 
    inquiry_series = df['inquiry_type'].value_counts()
    customer_position_series = df['customer_position'].value_counts()
    inquiry_replace_feature = []
    customer_replace_feature = []

    for idx, feature in enumerate(inquiry_series.index):
        if inquiry_series[idx] == 1:
            inquiry_replace_feature.append(feature)

    for idx, feature in enumerate(customer_position_series.index):
        if customer_position_series[idx] == 1:
            customer_replace_feature.append(feature)

    # 총 39개의 데이터 Concat
    df['inquiry_type'] = df['inquiry_type'].apply(lambda x: 'aimers_0203' if x in inquiry_replace_feature else x)
    # 총 53개의 데이터 Concat 
    df['customer_position'] = df['customer_position'].apply(lambda x: 'aimers_0203' if x in customer_replace_feature else x)

    return df
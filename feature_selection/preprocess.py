# preprocess 함수 사용할 것
# 예시 코드 (같은 폴더 내)
# from preprocess import preprocess
# df_train = preprocess(df_train)


# 열 삭제
# 'customer_country.1','product_subcategory','product_modelname','id_strategic_ver', 'it_strategic_ver', 'idit_strategic_ver'

def eda_drop_column(df):
    
    df = df.drop(['customer_country.1','product_subcategory','product_modelname','id_strategic_ver', 'it_strategic_ver', 'idit_strategic_ver'],axis=1)
    
    return df


# expected_timeline

def eda_expected_timeline(df):
    
    def timeline_label(time):
    
        time = str(time).lower().replace(' ','').replace('_','').replace('/','').replace(',','').replace('~','').replace('&','').replace('-','').replace('.','')
        
        if time == 'lessthan3months':
            result = 'less than 3 months'
        elif time == '3months6months':
            result = '3 months ~ 6 months'
        elif time == '6months9months':
            result = '6 months ~ 9 months'
        elif time == '9months1year':
            result = '9 months ~ 1 year'
        elif time == 'morethanayear':
            result = 'more than a year'
        else:
            result = 'aimer_0203'
            
        return result
    
    df['expected_timeline'] = df['expected_timeline'].apply(timeline_label)
    
    return df


# inquiry_type

def eda_inquiry_type_customer_position(df):
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
    # df['inquiry_type'] = df['inquiry_type'].apply(lambda x: 'aimers_0203' if x in inquiry_replace_feature else x)
    # 총 53개의 데이터 Concat 
    # df['customer_position'] = df['customer_position'].apply(lambda x: 'aimers_0203' if x in customer_replace_feature else x)

    return df


# business_area, business_subarea
# total_area 변수로 통일

def eda_business_area(df):
    
    for col in ['business_area','business_subarea']:
        
        df[col] = df[col].str.lower()
        df[col] = df[col].str.replace(" ", "") 
        df[col] = df[col].str.replace(r'[^\w\s]', "") 
        df[col] = df[col].fillna('nan') 
        
    df['total_area'] = df['business_area'].astype(str) + df['business_subarea'].astype(str)
    
    return df 


def preprocess(df):
    
    df = eda_drop_column(df)
    df = eda_expected_timeline(df)
    df = eda_inquiry_type_customer_position(df)
    df = eda_business_area(df)
    
    return df

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import re

df = pd.read_csv('credit_train.csv', sep = ';', encoding = 'windows-1251')

df.info()
print(df.describe())

# df['credit_sum'] = df['credit_sum'].replace(',', '', regex=True)
# df['credit_sum'] = df['credit_sum'].astype(float)

# df['score_shk'] = df['score_shk'].replace(',', '', regex=True)
# df['score_shk'] = df['score_shk'].astype(float)

# df = pd.get_dummies(df, columns = ['gender', 'marital_status', 'education'])

def clean_region(region):
    BAD_WORDS = ['РЕСПУБЛИКА', 'ОБЛАСТЬ', 'АВТОНОМНЫЙ ОКРУГ', 'ФЕДЕРАЛЬНЫЙ ОКРУГ', 'РЕСП', 'КРАЙ', 'ОБЛ', 'АОБЛ', ' АО', 'АО ', '.', ' Г', 'Г ', 'Р-Н',  '/']

    if pd.isnull(region):
        return None
    if not isinstance(region, str):
        return None
    
    for word in BAD_WORDS:
        region = re.sub(r'\b' + word + r'\b', '', region, flags=re.IGNORECASE)
    
    region = region.lower()
    return region.strip()
    

df['living_region'] = df['living_region'].apply(clean_region)



df.info()
print(df.describe())
print(df.value_counts('living_region'))
l = df['living_region'].values
print(set(l))
print(df)
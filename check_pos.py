import requests, io, pandas as pd
seasons = ['2016-17','2017-18','2018-19','2019-20']
BASE = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{s}/gws/merged_gw.csv'
for s in seasons:
    try:
        r = requests.get(BASE.format(s=s), timeout=20)
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        has_pos = 'position' in df.columns
        has_et  = 'element_type' in df.columns
        print(f"{s}: position={has_pos}, element_type={has_et}")
        if has_pos:
            print(f"  position unique: {list(df['position'].unique())[:10]}")
        if has_et:
            print(f"  element_type unique: {list(df['element_type'].unique())[:10]}")
    except Exception as e:
        print(f"{s}: ERROR {e}")

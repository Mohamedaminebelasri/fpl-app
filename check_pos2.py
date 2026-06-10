import requests, io, pandas as pd
seasons = ['2016-17','2017-18','2018-19','2019-20','2020-21']
BASE = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{s}/gws/merged_gw.csv'
for s in seasons:
    try:
        r = requests.get(BASE.format(s=s), timeout=20)
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        print(f"\n{s}: {df.shape}")
        print(f"  Cols: {list(df.columns)}")
    except Exception as e:
        print(f"{s}: ERROR {e}")

import json
from typing import Dict, List
import pandas as pd
from pyarrow import fs, csv
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio

# client = Minio(
#     '127.0.0.1:38001',
#     access_key="admin",
#     secret_key="8KW4cWXuGc",
#     secure=False,
#     cert_check=False
# )

# client.make_bucket(
#      'ko-en-mt-tech'
# )

# print(client.list_buckets())

minio = fs.S3FileSystem(
     endpoint_override='127.0.0.1:38001',
     access_key='admin',
     secret_key='8KW4cWXuGc',
     scheme='http')


json_paths = [
     'data/026.기술과학_분야_한-영_번역_병렬_말뭉치_데이터/01.데이터/1.Training/train_label.json',
     'data/026.기술과학_분야_한-영_번역_병렬_말뭉치_데이터/01.데이터/2.Validation/valid_label.json',
]


def load_json(_path):
     with open(_path) as f:
          return json.load(f)
     
path = 'ko-en-mt-tech/parquet'

def extract(_data : List[Dict]):
     target_keys = ['ko_original', 'en']
     result = {'ko': [], 'en': []}
     
     for x in _data:
          result['ko'].append(x[target_keys[0]])
          result['en'].append(x[target_keys[1]])
          
          
     return result
     
     

for _type, _path in zip(['train', 'valid'], json_paths):
     
     _json_data = load_json(_path)
     extracted = extract(_json_data['data'])

     train_pq = pa.Table.from_pydict(extracted)

     # convert the file to parquet and write it back to MinIO
     pq.write_to_dataset(
          table=train_pq,
          root_path=f'{path}/{_type}',
          filesystem=minio)
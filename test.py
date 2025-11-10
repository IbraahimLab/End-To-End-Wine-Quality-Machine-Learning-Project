import boto3
s3 = boto3.client('s3')
from wine_quality.constants import *

response = s3.list_objects_v2(Bucket='model-registry')
print(response)

# flow1 to mdoel

import openml
import sklearn
from sklearn_flow import to_flow
import requests
import jsonschema
import json

flow_id = 16347
flow = openml.flows.get_flow(flow_id)
s = openml.extensions.sklearn.extension.SklearnExtension()
model = s.flow_to_model(flow)
print(model)

flow2 = to_flow(model)
print(flow2)
jsonschema.validate(flow2, requests.get('https://openml.github.io/flow2/schemas/v0/pipeline.json').json())
print(json.dumps(flow2, allow_nan=False, ensure_ascii=False, indent=2))

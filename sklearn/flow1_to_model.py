# flow1 To mdoel

import openml
import sklearn
from sklearn_flow import to_flow
import requests
import jsonschema
import json



flow = openml.flows.get_flow(16347)
s = openml.extensions.sklearn.extension.SklearnExtension()
model = s.flow_to_model(flow)
print(model)
flow2 = to_flow(model)
jsonschema.validate(flow, requests.get('https://openml.github.io/flow2/schemas/v0/pipeline.json').json())
print(json.dumps(flow, allow_nan=False, ensure_ascii=False, indent=2))

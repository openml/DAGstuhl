import openml
import json
import jsonschema
import requests

f = openml.flows.get_flow(17360)
g = openml.flows.get_flow(8815)

a = f._to_dict()

flow1 = a['oml:flow']
print(json.dumps(flow1, allow_nan=False, ensure_ascii=False, indent=2))

flow2 = dict()
flow2['$schema']: str("http://json-schema.org/draft-04/schema")
flow2['id'] = flow1['oml:id']
flow2['uploader'] = flow1['oml:uploader']
flow2['name'] = flow1['oml:name']
flow2['version'] = flow1['oml:version']
flow2['external_version'] = flow1['oml:external_version']
flow2['description'] = flow1['oml:description']
flow2['upload_date'] = flow1['oml:upload_date']
flow2['language'] = flow1['oml:language']
flow2['dependencies'] = flow1['oml:dependencies']
flow2['class_name'] = flow1['oml:class_name']
# Making steps
# todo loop here after first entry
flow2['steps'] = []
for i in range(len(flow1['oml:component'])):
    flow2['steps'].append({})
    flow2['steps'][i]['type'] = 'SKLEARN'
    # Making estimator dict
    estimator = dict()
    estimator['python_path'] = flow1['oml:component'][i]['oml:flow']['oml:name']
    estimator['version'] = flow1['oml:component'][i]['oml:flow']['oml:external_version'] # todo: : trim version to int if really important
    flow2['steps'][i]['estimator'] = estimator
    # do not touch [oml:parameter] at all, its to messy
    flow2['steps'][i]['name'] = flow1['oml:component'][i]['oml:identifier']
    # Making hyperparams
    # todo: i need a loop here
    hyperparams = dict()
    params = flow1['oml:component'][i]['oml:flow']['oml:parameter']
    for j in range(len(params)):
        print(params[j])
        current_param = dict()
        current_param['type'] = 'VALUE'
        current_param['data'] = str(params[j]['oml:default_value'])
        hyperparams[params[j]['oml:name']] = current_param

    flow2['steps'][i]['hyperparams'] = hyperparams
    # Making argument parameter
    args = dict() # TODO
    # if i == 0:

flow2['tags'] = flow1['oml:tag']

print(json.dumps(flow2, allow_nan=False, ensure_ascii=False, indent=2))
jsonschema.validate(flow2, requests.get('https://openml.github.io/flow2/schemas/v0/pipeline.json').json())


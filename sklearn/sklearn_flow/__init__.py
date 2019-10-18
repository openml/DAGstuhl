import base64
import decimal
import importlib
import inspect
import numbers
import pickle
import typing
import uuid

import numpy

import sklearn
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer


def _is_sequence(value):
    return isinstance(value, typing.Sequence) and not isinstance(value, str)


def _feature_union_to_flow(pipeline_steps, transform, name, data_reference):
    params = transform.get_params(deep=False)
    transformers = params.pop('transformer_list')

    hyperparams = _encode_hyperparams(pipeline_steps, params)

    step_indices = []
    for transformer_name, transformer in transformers:
        step_indices.append(_transform_to_flow(pipeline_steps, transformer, transformer_name, None))

    hyperparams['transformer_list'] = {
        'type': 'STEP',
        'data': step_indices,
    }

    pipeline_step = _transform_step(transform, hyperparams, name, data_reference)

    pipeline_steps.append(pipeline_step)

    return len(pipeline_steps) - 1


def _column_transformer_to_flow(pipeline_steps, transform, name, data_reference):
    params = transform.get_params(deep=False)
    transformers = params.pop('transformers')

    params['transformer_columns'] = [columns for (transformer_name, transformer, columns) in transformers]

    hyperparams = _encode_hyperparams(pipeline_steps, params)

    step_indices = []
    for transformer_name, transformer, columns in transformers:
        step_indices.append(_transform_to_flow(pipeline_steps, transformer, transformer_name, None))

    hyperparams['transformers'] = {
        'type': 'STEP',
        'data': step_indices,
    }

    pipeline_step = _transform_step(transform, hyperparams, name, data_reference)

    pipeline_steps.append(pipeline_step)

    return len(pipeline_steps) - 1


def _pipeline_to_flow(pipeline_steps, transform, name, data_reference):
    pipeline_step = {
        'type': 'SUBPIPELINE',
        'pipeline': to_flow(transform),
    }

    if name:
        pipeline_step['name'] = name

    if data_reference is not None:
        pipeline_step.update({
            'arguments': {
                'input': {
                    'type': 'CONTAINER',
                    'data': data_reference,
                },
            },
            'outputs': [{
                'id': 'output',
            }],
        })

    pipeline_steps.append(pipeline_step)

    return len(pipeline_steps) - 1


def _encode_hyperparameter_value(parameter_value):
    if _is_sequence(parameter_value):
        return [_encode_hyperparameter_value(value) for value in parameter_value]
    elif isinstance(parameter_value, (str, int, bool, type(None), numpy.integer, numbers.Integral)):
        return parameter_value
    elif isinstance(parameter_value, (float, numpy.float32, numpy.float64, decimal.Decimal, numbers.Real)):
        if not numpy.isfinite(parameter_value):
            return {
                'encoding': 'pickle',
                'value': base64.b64encode(pickle.dumps(parameter_value)).decode('utf8'),
            }
        else:
            return parameter_value
    else:
        return {
            'encoding': 'pickle',
            'value': base64.b64encode(pickle.dumps(parameter_value)).decode('utf8'),
        }


def _is_unfitted_estimator(value):
    if not isinstance(value, BaseEstimator):
        return False

    if 'deep' in inspect.signature(value.get_params).parameters:
        params = value.get_params(deep=False)
    else:
        params = value.get_params()

    for parameter_name in params.keys():
        if parameter_name.endswith('_'):
            return False

    return True


def _encode_hyperparams(pipeline_steps, params):
    hyperparams = {}
    for parameter_name, parameter_value in params.items():
        if parameter_name.startswith('_'):
            continue
        if parameter_name.endswith('_'):
            raise ValueError(f"Encountered an already fitted estimator.")

        if _is_unfitted_estimator(parameter_value):
            hyperparams[parameter_name] = {
                'type': 'STEP',
                'data': _transform_to_flow(pipeline_steps, parameter_value, None, None),
            }
        elif _is_sequence(parameter_value) and all(_is_unfitted_estimator(v) for v in parameter_value):
            hyperparams[parameter_name] = {
                'type': 'STEP',
                'data': [_transform_to_flow(pipeline_steps, v, None, None) for v in parameter_value],
            }
        else:
            # We cannot really know which parameters can be tuned and which cannot,
            # so we put all of them into the flow.
            hyperparams[parameter_name] = {
                'type': 'VALUE',
                'data': _encode_hyperparameter_value(parameter_value),
            }

    return hyperparams


def _transform_step(transform, hyperparams, name, data_reference):
    transform_class = type(transform)
    pipeline_step = {
        'type': 'SKLEARN',
        'estimator': {
            'python_path': f'{transform_class.__module__}.{transform_class.__name__}',
            'version': sklearn.__version__,
        },
    }

    if name:
        pipeline_step['name'] = name

    if hyperparams:
        pipeline_step['hyperparams'] = hyperparams

    if data_reference is not None:
        pipeline_step.update({
            'arguments': {
                'input': {
                    'type': 'CONTAINER',
                    'data': data_reference,
                },
            },
            'outputs': [{
                'id': 'output',
            }],
        })

    return pipeline_step


def _transform_to_flow(pipeline_steps, transform, name, data_reference):
    if isinstance(transform, Pipeline):
        return _pipeline_to_flow(pipeline_steps, transform, name, data_reference)
    elif isinstance(transform, FeatureUnion):
        return _feature_union_to_flow(pipeline_steps, transform, name, data_reference)
    elif isinstance(transform, ColumnTransformer):
        return _column_transformer_to_flow(pipeline_steps, transform, name, data_reference)
    else:
        pipeline_step = _transform_step(transform, _encode_hyperparams(pipeline_steps, transform.get_params()), name, data_reference)

        pipeline_steps.append(pipeline_step)

        return len(pipeline_steps) - 1


def to_flow(sklearn_pipeline):
    pipeline_steps = []

    current_data_reference = 'inputs.0'
    for name, transform in sklearn_pipeline.steps:
        main_step_index = _transform_to_flow(pipeline_steps, transform, name, current_data_reference)
        current_data_reference = f'steps.{main_step_index}.output'

    return {
        'schema': 'https://openml.github.io/flow2/schemas/v0/pipeline.json',
        'id': str(uuid.uuid4()),
        'inputs': [{
            'name': 'pipeline input',
        }],
        'outputs': [{
            'name': 'pipeline output',
            'data': current_data_reference,
        }],
        'steps': pipeline_steps,
    }


def _decode_hyperparameter_value(hyperparameter_value):
    if _is_sequence(hyperparameter_value):
        return [_decode_hyperparameter_value(value) for value in hyperparameter_value]
    elif isinstance(hyperparameter_value, typing.Mapping):
        if hyperparameter_value.get('encoding', None) == 'pickle':
            return pickle.loads(base64.b64decode(hyperparameter_value['value'].encode('utf8')))
        else:
            raise ValueError(f"Invalid hyper-parameter value encoding: {hyperparameter_value.get('encoding', None)}")
    else:
        return hyperparameter_value


def _decode_hyperparams(flow_steps, hyperparams):
    params = {}
    for hyperparameter_name, hyperparameter in hyperparams.items():
        if hyperparameter['type'] == 'VALUE':
            params[hyperparameter_name] = _decode_hyperparameter_value(hyperparameter['data'])
        elif hyperparameter['type'] == 'STEP':
            if _is_sequence(hyperparameter['data']):
                params[hyperparameter_name] = [_transform_from_flow_step(flow_steps, flow_steps[step_index]) for step_index in hyperparameter['data']]
            else:
                params[hyperparameter_name] = _transform_from_flow_step(flow_steps, flow_steps[hyperparameter['data']])
        else:
            raise ValueError(f"Invalid hyper-parameter type: {hyperparameter['type']}")

    return params


def _feature_union_from_flow(flow_steps, step, transform_class):
    params = _decode_hyperparams(flow_steps, step['hyperparams'])

    params['transformer_list'] = [
        (flow_steps[step_index]['name'], transformer)
        for step_index, transformer
        in zip(step['hyperparams']['transformer_list']['data'], params['transformer_list'])
    ]

    return _transform_instance(transform_class, params)


def _column_transformer_from_flow(flow_steps, step, transform_class):
    params = _decode_hyperparams(flow_steps, step['hyperparams'])

    transformer_columns = params.pop('transformer_columns')

    params['transformers'] = [
        (flow_steps[step_index]['name'], transformer, columns)
        for step_index, transformer, columns
        in zip(step['hyperparams']['transformers']['data'], params['transformers'], transformer_columns)
    ]

    return _transform_instance(transform_class, params)


def _transform_instance(transform_class, params):
    return transform_class(**params)


def _transform_from_class(flow_steps, step, transform_class):
    if issubclass(transform_class, FeatureUnion):
        return _feature_union_from_flow(flow_steps, step, transform_class)
    elif issubclass(transform_class, ColumnTransformer):
        return _column_transformer_from_flow(flow_steps, step, transform_class)
    else:
        return _transform_instance(transform_class, _decode_hyperparams(flow_steps, step['hyperparams']))


def _transform_from_flow_step(flow_steps, step):
    if step['type'] == 'SKLEARN':
        python_path = step['estimator']['python_path']
        (module_path, _, class_name) = python_path.rpartition('.')
        module_ = importlib.import_module(module_path)
        transform_class = getattr(module_, class_name)
        return _transform_from_class(flow_steps, step, transform_class)
    elif step['type'] == 'SUBPIPELINE':
        return from_flow(step['pipeline'])
    else:
        raise ValueError(f"Invalid step type: {step['type']}")


def from_flow(flow_pipeline):
    pipeline_steps = []

    if len(flow_pipeline['inputs']) != 1:
        raise ValueError(f"Invalid number of pipeline inputs: {len(flow_pipeline['inputs'])}")
    if len(flow_pipeline['outputs']) != 1:
        raise ValueError(f"Invalid number of pipeline outputs: {len(flow_pipeline['outputs'])}")

    current_data_reference = 'inputs.0'
    for i, step in enumerate(flow_pipeline['steps']):
        arguments = step.get('arguments', {})
        if arguments:
            if set(arguments.keys()) == {'input'}:
                if arguments['input']['type'] != 'CONTAINER':
                    raise ValueError(f"Invalid input type for step {i}: {arguments['input']['type']}")
                if arguments['input']['data'] != current_data_reference:
                    raise ValueError(f"Expected input data reference '{current_data_reference}' does not match provided input data reference '{arguments['input']['data']}' for step {i}.")
            else:
                raise ValueError(f"Invalid step arguments for step {i}: {sorted(set(arguments.keys()))}")

        outputs = step.get('outputs', {})
        if outputs:
            if len(outputs) == 1:
                if outputs[0]['id'] != 'output':
                    raise ValueError(f"Invalid output data id for step {i}: {outputs[0]['id']}")
            else:
                raise ValueError(f"Invalid number of step outputs for step {i}: {len(outputs)}")

        if arguments and outputs:
            if 'name' not in step:
                raise ValueError(f"Missing step name for step {i}.")

            pipeline_steps.append((step['name'], _transform_from_flow_step(flow_pipeline['steps'], step)))

            current_data_reference = f'steps.{i}.output'

    if current_data_reference != flow_pipeline['outputs'][0]['data']:
        raise ValueError(f"Expected output data reference '{current_data_reference}' does not match provided output data reference '{flow_pipeline['outputs'][0]['data']}'.")

    return Pipeline(steps=pipeline_steps)

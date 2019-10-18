import base64
import decimal
import numbers
import pickle
import uuid

import numpy

import sklearn
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer


def _feature_union_to_flow(pipeline_steps, transform, name, data_reference):
    params = transform.get_params(deep=False)
    transformers = params.pop('transformer_list')

    hyperparams = _encode_hyperparams(transform, params)

    step_indices = []
    for transformer_name, transformer, columns in transformers:
        step_indices.append(_transform_to_flow(pipeline_steps, transformer, transformer_name, None))

    hyperparams['transformer_list'] = {
        'type': 'STEP',
        'data': step_indices,
    }

    pipeline_step = _transform(transform, hyperparams, name, data_reference)

    pipeline_steps.append(pipeline_step)

    return len(pipeline_steps) - 1


def _column_transformer_to_flow(pipeline_steps, transform, name, data_reference):
    params = transform.get_params(deep=False)
    transformers = params.pop('transformers')

    params['transformer_columns'] = [columns for (transformer_name, transformer, columns) in transformers]

    hyperparams = _encode_hyperparams(transform, params)

    step_indices = []
    for transformer_name, transformer, columns in transformers:
        step_indices.append(_transform_to_flow(pipeline_steps, transformer, transformer_name, None))

    hyperparams['transformers'] = {
        'type': 'STEP',
        'data': step_indices,
    }

    pipeline_step = _transform(transform, hyperparams, name, data_reference)

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
    if isinstance(parameter_value, list):
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


def _encode_hyperparams(transform, params):
    hyperparams = {}
    for parameter_name, parameter_value in params.items():
        if parameter_name.startswith('_'):
            continue
        if parameter_name.endswith('_'):
            raise ValueError(f"Encountered an already fitted estimator: {transform}")

        # We cannot really know which parameters can be tuned and which cannot,
        # so we put all of them into the flow.
        hyperparams[parameter_name] = {
            'type': 'VALUE',
            'data': _encode_hyperparameter_value(parameter_value),
        }

    return hyperparams


def _transform(transform, hyperparams, name, data_reference):
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
        pipeline_step = _transform(transform, _encode_hyperparams(transform, transform.get_params()), name, data_reference)

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


def from_flow(flow_pipeline):
    pass

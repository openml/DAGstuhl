# Flow 2 pipeline language

## Schema

See [https://openml.github.io/flow2/schemas/v0/pipeline.json](https://openml.github.io/flow2/schemas/v0/pipeline.json)
and [https://openml.github.io/flow2/schemas/v0/definitions.json](https://openml.github.io/flow2/schemas/v0/definitions.json)
JSON schemas defining the schema of valid pipelines.

The schema defines the structure of schema when a pipeline is represented in its JSON-compatible
structure. How those JSON-compatible structures are serialized/stored/transmitted depends on what
is most suitable for a concrete use case (JSON, YAML, ProtoBuf, etc.).

## Comments on the design

* The pipeline language tries to strike the right balance between complexity and generality. It is
  a simple DAG description language where steps are not fully specified but are left to a particular
  programming language and runtime to define. Encoding of hyper-parameters is left of this language as well.
* Similarly, when designing a pipeline language the question is how much logic can connections themselves
  have. Do we want to support some basic operations on values as they are passed between steps. This
  pipeline language provides a narrow support here with idea that most other operations should be described
  as steps themselves.
* The pipeline language allows additional non-standard properties everywhere. Even in cases where just a list
  of strings would generally be enough, we use a list of dicts so that additional properties can be added
  if a particular programming language or runtime wants to add more information.
* The pipeline language connects outputs to inputs using *data references*. Those data references can represent
  output values to be connected, but also connecting underlying step implementations themselves. The latter
  provides support for higher-order steps which take other steps as inputs.
* The pipeline language does not define the execution semantics of a pipeline, just its structure.
  A particular programming language and/or runtime used define that.

## Types of connections

Arguments and hyper-parameters together form inputs to a step. The pipeline language on purpose supports only
a limited number of possible connection types:

* a `CONTAINER` value: container values are those which consist of more underlying values (lists, ndarrays, etc.)
* a list of `CONTAINER` values: variable number of `CONTAINER` connections to the same input of a step is expressed as a list of container values
* a `DATA` value: is one unit of a `CONTAINER` value (element of a list, but also a row of an ndarray)
* a list of `DATA` values: variable number of `DATA` connections to the same input of a step is expressed as a list of data values
* a `VALUE` is just a constant
* a `STEP` is passing the underlying instance of step, used to define higher-order steps
* a list of `STEP`s: variable number of underlying instances of steps

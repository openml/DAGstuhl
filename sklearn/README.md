# scikit-learn pipeline conversion

Here you can find a prototype of a scikit-learn pipeline conversion to the flow 2
pipeline language.

## Example

First install [`requirements.txt`](../requirements.txt).

```bash
$ ./to_flow_example.py > example-pipeline.json
$ cat example-pipeline.json | ./from_flow_example.py
model score: 0.790
```

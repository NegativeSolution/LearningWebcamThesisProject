# Requires: pip install onnx numpy
import onnx
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python inspect_onnx.py <model.onnx>")
    sys.exit(1)

p = Path(sys.argv[1])
m = onnx.load_model(str(p))
g = m.graph

print("Model:", p.name)
print("\nOutputs:")
for o in g.output:
    shape = []
    for d in o.type.tensor_type.shape.dim:
        shape.append(d.dim_value if d.dim_value else '?')
    print(f" - {o.name} : shape={shape}")

print("\nInitializers (weights/biases):")
for init in g.initializer:
    dims = list(init.dims)
    print(f" - {init.name} : dims={dims} dtype={init.data_type}")

print("\nNodes (last 20):")
for node in g.node[-20:]:
    print(f" - op:{node.op_type} outputs:{list(node.output)} inputs:{list(node.input)}")
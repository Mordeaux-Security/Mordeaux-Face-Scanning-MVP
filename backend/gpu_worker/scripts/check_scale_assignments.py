"""Check scale assignments for specific tensors."""

import onnx
import os

model_path = r'C:\Users\Davis\.insightface\models\buffalo_l\det_10g_modified.onnx'
model = onnx.load(model_path)

# Check tensor 467 (input to Transpose_198)
tensor_467 = next((vi for vi in model.graph.value_info if vi.name == '467'), None)
if tensor_467:
    shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in tensor_467.type.tensor_type.shape.dim]
    print(f"Tensor 467: {shape}")
    if len(shape) >= 4:
        h_param = shape[2]
        w_param = shape[3]
        print(f"  Spatial parameters: H={h_param}, W={w_param}")

# Find all tensors with the same spatial parameters as 467
if tensor_467 and tensor_467.type.tensor_type.shape and len(tensor_467.type.tensor_type.shape.dim) >= 4:
    h_param = tensor_467.type.tensor_type.shape.dim[2].dim_param if tensor_467.type.tensor_type.shape.dim[2].HasField('dim_param') else None
    w_param = tensor_467.type.tensor_type.shape.dim[3].dim_param if tensor_467.type.tensor_type.shape.dim[3].HasField('dim_param') else None
    
    if h_param and w_param:
        print(f"\nFinding all tensors with H={h_param}, W={w_param}:")
        matching_tensors = []
        for vi in model.graph.value_info:
            if vi.type.tensor_type.shape and len(vi.type.tensor_type.shape.dim) >= 4:
                vi_h = vi.type.tensor_type.shape.dim[2].dim_param if vi.type.tensor_type.shape.dim[2].HasField('dim_param') else None
                vi_w = vi.type.tensor_type.shape.dim[3].dim_param if vi.type.tensor_type.shape.dim[3].HasField('dim_param') else None
                if vi_h == h_param and vi_w == w_param:
                    matching_tensors.append(vi.name)
        
        print(f"  Found {len(matching_tensors)} tensors:")
        for tname in matching_tensors[:20]:
            print(f"    - {tname}")
        if len(matching_tensors) > 20:
            print(f"    ... and {len(matching_tensors) - 20} more")

# Check what feeds into tensor 467
print(f"\nTracing back from tensor 467:")
producers = {}
for node in model.graph.node:
    for output in node.output:
        producers[output] = node

tensor_name = '467'
depth = 0
max_depth = 5

def trace_back(tname, depth):
    if depth > max_depth:
        return
    if tname in producers:
        producer = producers[tname]
        print(f"  {'  ' * depth}{producer.name} ({producer.op_type}) -> {tname}")
        for inp in producer.input:
            if inp and inp != tname:
                trace_back(inp, depth + 1)

trace_back('467', 0)




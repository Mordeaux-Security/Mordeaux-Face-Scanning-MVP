"""Compare original and modified model shapes for Resize nodes."""

import onnx
import os

orig_path = r'C:\Users\Davis\.insightface\models\buffalo_l\det_10g.onnx'
mod_path = r'C:\Users\Davis\.insightface\models\buffalo_l\det_10g_modified.onnx'

orig = onnx.load(orig_path)
mod = onnx.load(mod_path)

print("Comparing Resize node inputs/outputs:")
print("=" * 80)

nodes_to_check = [('Resize_124', '382', '399', '401'), ('Resize_144', '402', '419', '421')]

for node_name, data_input, sizes_input, output in nodes_to_check:
    print(f"\n{node_name}:")
    print(f"  Data input: {data_input}")
    print(f"  Sizes input: {sizes_input}")
    print(f"  Output: {output}")
    
    # Check data input
    orig_data_vi = next((vi for vi in orig.graph.value_info if vi.name == data_input), None)
    mod_data_vi = next((vi for vi in mod.graph.value_info if vi.name == data_input), None)
    
    if orig_data_vi:
        orig_shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in orig_data_vi.type.tensor_type.shape.dim]
        print(f"  Original data shape: {orig_shape}")
    if mod_data_vi:
        mod_shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in mod_data_vi.type.tensor_type.shape.dim]
        print(f"  Modified data shape: {mod_shape}")
    
    # Check sizes input
    orig_sizes_vi = next((vi for vi in orig.graph.value_info if vi.name == sizes_input), None)
    mod_sizes_vi = next((vi for vi in mod.graph.value_info if vi.name == sizes_input), None)
    
    if orig_sizes_vi:
        orig_shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in orig_sizes_vi.type.tensor_type.shape.dim]
        print(f"  Original sizes shape: {orig_shape}")
    if mod_sizes_vi:
        mod_shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in mod_sizes_vi.type.tensor_type.shape.dim]
        print(f"  Modified sizes shape: {mod_shape}")
    
    # Check output
    orig_out_vi = next((vi for vi in orig.graph.value_info if vi.name == output), None)
    mod_out_vi = next((vi for vi in mod.graph.value_info if vi.name == output), None)
    
    if orig_out_vi:
        orig_shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in orig_out_vi.type.tensor_type.shape.dim]
        print(f"  Original output shape: {orig_shape}")
    if mod_out_vi:
        mod_shape = [d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in mod_out_vi.type.tensor_type.shape.dim]
        print(f"  Modified output shape: {mod_shape}")

print("\n" + "=" * 80)
print("Key Difference:")
print("Original model: Batch dimension is '?' (dynamic) throughout")
print("Modified model: Batch dimension is 32 (fixed) throughout")
print("This might cause ONNX Runtime to be more aggressive with buffer reuse,")
print("leading to conflicts when different scales have different spatial dimensions.")




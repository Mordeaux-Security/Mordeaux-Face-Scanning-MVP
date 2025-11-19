"""
Inspect Original Model Dynamic Dimension Usage

Analyzes how the original SCRFD model defines dynamic dimensions across tensors
to understand how it avoids buffer reuse mismatches in multi-scale architectures.
"""

import onnx
import os
import sys
import io
from collections import defaultdict

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def analyze_dynamic_dimensions(model_path: str):
    """Analyze how the original model defines dynamic dimensions."""
    
    print("=" * 80)
    print("Original Model Dynamic Dimension Analysis")
    print("=" * 80)
    print(f"Model: {model_path}")
    print()
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    print("📥 Loading model...")
    model = onnx.load(model_path)
    
    # Analyze input
    print("\n" + "=" * 80)
    print("INPUT ANALYSIS")
    print("=" * 80)
    input_tensor = model.graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape
    print(f"Input: {input_tensor.name}")
    print(f"Shape: {[d.dim_value if d.HasField('dim_value') else (d.dim_param if d.HasField('dim_param') else '?') for d in input_shape.dim]}")
    print()
    
    # Analyze all tensors with dynamic dimensions
    print("=" * 80)
    print("DYNAMIC DIMENSION ANALYSIS")
    print("=" * 80)
    
    # Track dimension parameters used
    dim_params = defaultdict(list)  # param_name -> list of (tensor_name, dim_index, full_shape)
    tensor_shapes = {}  # tensor_name -> shape
    
    # Check inputs
    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        shape_list = []
        for i, dim in enumerate(shape.dim):
            if dim.HasField('dim_param'):
                param = dim.dim_param
                dim_params[param].append((inp.name, i, None))
                shape_list.append(param)
            elif dim.HasField('dim_value'):
                shape_list.append(dim.dim_value)
            else:
                shape_list.append('?')
        tensor_shapes[inp.name] = shape_list
    
    # Check value_info (intermediate tensors)
    for vi in model.graph.value_info:
        if not vi.type.tensor_type.shape:
            continue
        shape = vi.type.tensor_type.shape
        shape_list = []
        for i, dim in enumerate(shape.dim):
            if dim.HasField('dim_param'):
                param = dim.dim_param
                dim_params[param].append((vi.name, i, None))
                shape_list.append(param)
            elif dim.HasField('dim_value'):
                shape_list.append(dim.dim_value)
            else:
                shape_list.append('?')
        tensor_shapes[vi.name] = shape_list
    
    # Check outputs
    for out in model.graph.output:
        shape = out.type.tensor_type.shape
        shape_list = []
        for i, dim in enumerate(shape.dim):
            if dim.HasField('dim_param'):
                param = dim.dim_param
                dim_params[param].append((out.name, i, None))
                shape_list.append(param)
            elif dim.HasField('dim_value'):
                shape_list.append(dim.dim_value)
            else:
                shape_list.append('?')
        tensor_shapes[out.name] = shape_list
    
    # Now update dim_params with full shapes
    for param_name, tensor_list in dim_params.items():
        for i, (tensor_name, dim_idx, _) in enumerate(tensor_list):
            if tensor_name in tensor_shapes:
                dim_params[param_name][i] = (tensor_name, dim_idx, tensor_shapes[tensor_name])
    
    # Print analysis
    print(f"\nFound {len(dim_params)} unique dimension parameters:")
    print()
    
    for param_name, tensor_list in sorted(dim_params.items()):
        print(f"📊 Parameter: '{param_name}'")
        print(f"   Used in {len(tensor_list)} tensor(s) at dimension index(es):")
        
        # Group by dimension index
        by_dim_idx = defaultdict(list)
        for tensor_name, dim_idx, full_shape in tensor_list:
            by_dim_idx[dim_idx].append((tensor_name, full_shape))
        
        for dim_idx in sorted(by_dim_idx.keys()):
            tensors = by_dim_idx[dim_idx]
            print(f"   Dimension {dim_idx}:")
            for tensor_name, full_shape in tensors[:10]:  # Show first 10
                shape_str = str(full_shape) if full_shape else 'unknown'
                print(f"      - {tensor_name}: {shape_str}")
            if len(tensors) > 10:
                print(f"      ... and {len(tensors) - 10} more")
        print()
    
    # Analyze spatial dimensions specifically (4D tensors)
    print("=" * 80)
    print("SPATIAL DIMENSION ANALYSIS (4D Tensors)")
    print("=" * 80)
    
    spatial_dims = {}  # tensor_name -> (batch_dim, channel_dim, h_dim, w_dim)
    
    for tensor_name, shape in tensor_shapes.items():
        if len(shape) == 4:  # 4D tensor (N, C, H, W)
            batch_dim = shape[0]
            channel_dim = shape[1]
            h_dim = shape[2]
            w_dim = shape[3]
            spatial_dims[tensor_name] = (batch_dim, channel_dim, h_dim, w_dim)
    
    print(f"\nFound {len(spatial_dims)} 4D tensors (image-like activations):")
    print()
    
    # Group by spatial dimension pattern
    spatial_patterns = defaultdict(list)
    for tensor_name, (b, c, h, w) in spatial_dims.items():
        # Create pattern key based on spatial dimensions
        spatial_key = f"H={h}, W={w}"
        spatial_patterns[spatial_key].append((tensor_name, b, c))
    
    print("Spatial dimension patterns:")
    for pattern, tensors in sorted(spatial_patterns.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n  {pattern} ({len(tensors)} tensors):")
        for tensor_name, b, c in tensors[:10]:  # Show first 10
            print(f"    - {tensor_name}: batch={b}, channels={c}")
        if len(tensors) > 10:
            print(f"    ... and {len(tensors) - 10} more")
    
    # Check for Resize nodes and their inputs/outputs
    print("\n" + "=" * 80)
    print("RESIZE NODE ANALYSIS")
    print("=" * 80)
    
    resize_nodes = [(i, node) for i, node in enumerate(model.graph.node) if node.op_type == 'Resize']
    print(f"\nFound {len(resize_nodes)} Resize nodes:")
    
    for i, (node_idx, node) in enumerate(resize_nodes[:5]):  # Show first 5
        print(f"\n  Resize node {i+1}: {node.name} (index {node_idx})")
        print(f"    Inputs: {node.input}")
        print(f"    Outputs: {node.output}")
        
        # Check input shapes
        for inp_name in node.input:
            if inp_name in tensor_shapes:
                print(f"    Input '{inp_name}': {tensor_shapes[inp_name]}")
        
        # Check output shapes
        for out_name in node.output:
            if out_name in tensor_shapes:
                print(f"    Output '{out_name}': {tensor_shapes[out_name]}")
    
    if len(resize_nodes) > 5:
        print(f"\n  ... and {len(resize_nodes) - 5} more Resize nodes")
    
    # Key insight: Check if different scales use different dimension parameters
    print("\n" + "=" * 80)
    print("MULTI-SCALE DIMENSION PARAMETER ANALYSIS")
    print("=" * 80)
    
    # Find tensors that likely belong to different scales based on spatial dimensions
    # Group 4D tensors by their spatial dimension values/parameters
    scale_groups = defaultdict(list)
    
    for tensor_name, (b, c, h, w) in spatial_dims.items():
        # Create a scale identifier based on spatial dimensions
        # If both H and W are the same dynamic param or value, they're likely the same scale
        if isinstance(h, str) and isinstance(w, str):
            if h == w:  # Same parameter for H and W
                scale_key = f"scale_{h}"
            else:
                scale_key = f"scale_H{h}_W{w}"
        elif isinstance(h, int) and isinstance(w, int):
            if h == w:
                scale_key = f"scale_{h}x{h}"
            else:
                scale_key = f"scale_{h}x{w}"
        else:
            scale_key = f"scale_mixed_H{h}_W{w}"
        
        scale_groups[scale_key].append((tensor_name, b, c, h, w))
    
    print(f"\nIdentified {len(scale_groups)} potential scale groups:")
    for scale_key, tensors in sorted(scale_groups.items()):
        print(f"\n  {scale_key} ({len(tensors)} tensors):")
        for tensor_name, b, c, h, w in tensors[:5]:
            print(f"    - {tensor_name}: batch={b}, channels={c}, H={h}, W={w}")
        if len(tensors) > 5:
            print(f"    ... and {len(tensors) - 5} more")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Check if the original model uses the same or different dim_params for different scales
    print("\n1. Dimension Parameter Usage:")
    if len(dim_params) == 1 and list(dim_params.keys())[0] in ('?', None):
        print("   ⚠️  All dynamic dimensions use the same parameter - potential issue!")
    else:
        print(f"   ✅ Model uses {len(dim_params)} different dimension parameters")
        print("   This allows different scales to have different spatial dimensions")
    
    print("\n2. Spatial Dimension Patterns:")
    print(f"   Found {len(spatial_patterns)} different spatial dimension patterns")
    print("   This indicates multi-scale feature extraction")
    
    print("\n3. Scale Groups:")
    print(f"   Identified {len(scale_groups)} potential scale groups")
    if len(scale_groups) > 1:
        print("   ✅ Multiple scales detected - model uses multi-scale architecture")
        print("   ⚠️  Need to ensure each scale uses appropriate dimension parameters")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)

def main():
    """Main function."""
    # Find original model
    home = os.path.expanduser("~/.insightface")
    model_path = os.path.join(home, "models", "buffalo_l", "det_10g.onnx")
    
    if not os.path.exists(model_path):
        print(f"❌ Original model not found: {model_path}")
        print("\nPlease ensure InsightFace models are installed.")
        sys.exit(1)
    
    analyze_dynamic_dimensions(model_path)

if __name__ == "__main__":
    main()




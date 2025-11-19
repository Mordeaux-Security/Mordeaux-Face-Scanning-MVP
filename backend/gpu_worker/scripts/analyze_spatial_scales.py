"""
Analyze Spatial Scales in ONNX Model

Identifies different spatial scales in a multi-scale architecture by analyzing
downsampling operations, spatial dimension relationships, and tensor shapes.
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

def analyze_downsampling_operations(model: onnx.ModelProto) -> dict:
    """
    Analyze downsampling operations to identify scale changes.
    
    Returns:
        Dictionary mapping tensor names to their scale ratios relative to input
    """
    print("🔍 Analyzing downsampling operations...")
    
    # Get input spatial dimensions (assuming 640x640 after letterbox)
    input_tensor = model.graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape
    # Input is typically [1, 3, '?', '?'] - we'll assume 640x640 at runtime
    
    # Build tensor dependency graph
    tensor_producers = {}  # tensor_name -> node that produces it
    tensor_shape_info = {}  # tensor_name -> shape info
    
    # Collect shape information
    for vi in model.graph.value_info:
        if vi.type.tensor_type.shape:
            tensor_shape_info[vi.name] = vi.type.tensor_type.shape
    
    for inp in model.graph.input:
        if inp.type.tensor_type.shape:
            tensor_shape_info[inp.name] = inp.type.tensor_type.shape
    
    for out in model.graph.output:
        if out.type.tensor_type.shape:
            tensor_shape_info[out.name] = out.type.tensor_type.shape
    
    # Build producer map
    for node in model.graph.node:
        for output in node.output:
            tensor_producers[output] = node
    
    # Track scale ratios (relative to input 640x640)
    # We'll use a heuristic: track downsampling through stride operations
    tensor_scale = {}  # tensor_name -> scale factor (1.0 = input scale, 0.5 = half scale, etc.)
    
    # Input is at scale 1.0 (640x640)
    input_name = input_tensor.name
    tensor_scale[input_name] = 1.0
    
    # Process nodes in topological order
    # We'll need to process nodes multiple times until convergence
    changed = True
    iterations = 0
    max_iterations = 100
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        for node in model.graph.node:
            op_type = node.op_type
            
            # Skip if we don't know the scale of any input
            input_scales = []
            for inp in node.input:
                if inp in tensor_scale:
                    input_scales.append(tensor_scale[inp])
            
            if not input_scales:
                continue
            
            # Default: output scale = input scale
            output_scale = input_scales[0]
            
            # Handle downsampling operations
            if op_type == 'Conv':
                # Check stride attribute
                stride_h = 1
                stride_w = 1
                for attr in node.attribute:
                    if attr.name == 'strides':
                        if attr.type == 7:  # INTS
                            strides = list(attr.ints)
                            if len(strides) >= 2:
                                stride_h = strides[0]
                                stride_w = strides[1]
                
                # Downsampling factor
                output_scale = input_scales[0] / max(stride_h, stride_w)
                
            elif op_type == 'MaxPool' or op_type == 'AveragePool':
                # Check stride attribute
                stride_h = 1
                stride_w = 1
                for attr in node.attribute:
                    if attr.name == 'strides':
                        if attr.type == 7:  # INTS
                            strides = list(attr.ints)
                            if len(strides) >= 2:
                                stride_h = strides[0]
                                stride_w = strides[1]
                
                output_scale = input_scales[0] / max(stride_h, stride_w)
                
            elif op_type == 'Resize':
                # Resize operations create new scales
                # We can't determine the exact scale from attributes alone,
                # but we can identify that it's different
                # Use a unique scale identifier
                output_scale = input_scales[0]  # Will be adjusted based on sizes input
                
                # Check if sizes input gives us information
                if len(node.input) >= 4:
                    sizes_input = node.input[3]
                    # Try to get scale from sizes if it's a constant
                    # (This is complex, so we'll use a heuristic)
                    pass
            
            # Update output scales
            for output in node.output:
                if output not in tensor_scale or tensor_scale[output] != output_scale:
                    tensor_scale[output] = output_scale
                    changed = True
    
    print(f"   Identified scales for {len(tensor_scale)} tensors after {iterations} iterations")
    
    # Group by scale factor
    scale_groups = defaultdict(list)
    for tensor_name, scale in tensor_scale.items():
        # Round to nearest reasonable scale factor
        if scale >= 0.9:
            scale_group = 1.0  # Full scale
        elif scale >= 0.4:
            scale_group = 0.5  # Half scale
        elif scale >= 0.2:
            scale_group = 0.25  # Quarter scale
        elif scale >= 0.1:
            scale_group = 0.125  # Eighth scale
        else:
            scale_group = scale  # Keep as-is
        
        scale_groups[scale_group].append(tensor_name)
    
    print(f"   Found {len(scale_groups)} distinct scale groups:")
    for scale_factor in sorted(scale_groups.keys()):
        print(f"     Scale {scale_factor}: {len(scale_groups[scale_factor])} tensors")
    
    return tensor_scale, scale_groups

def identify_scales_from_resize_outputs(model: onnx.ModelProto) -> dict:
    """
    Identify scales by analyzing Resize node outputs and their spatial dimensions.
    
    This is a more direct approach: Resize nodes create explicit scale changes.
    """
    print("🔍 Identifying scales from Resize operations and spatial relationships...")
    
    # Find all Resize nodes
    resize_nodes = [node for node in model.graph.node if node.op_type == 'Resize']
    print(f"   Found {len(resize_nodes)} Resize nodes")
    
    # Build tensor shape map
    tensor_shapes = {}
    for vi in model.graph.value_info:
        if vi.type.tensor_type.shape:
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            tensor_shapes[vi.name] = shape
    
    # Also get input shape
    input_tensor = model.graph.input[0]
    if input_tensor.type.tensor_type.shape:
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            elif dim.HasField('dim_param'):
                shape.append(dim.dim_param)
            else:
                shape.append('?')
        tensor_shapes[input_tensor.name] = shape
    
    # Analyze Resize nodes to understand scale relationships
    resize_scales = {}  # resize_output_name -> scale_info
    for resize_node in resize_nodes:
        if not resize_node.output:
            continue
        
        output_name = resize_node.output[0]
        input_name = resize_node.input[0] if resize_node.input else None
        
        # Get shapes
        input_shape = tensor_shapes.get(input_name, [])
        output_shape = tensor_shapes.get(output_name, [])
        
        print(f"   Resize '{resize_node.name}':")
        print(f"     Input: {input_name} -> {input_shape}")
        print(f"     Output: {output_name} -> {output_shape}")
        
        resize_scales[output_name] = {
            'input': input_name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'node': resize_node
        }
    
    return resize_scales, tensor_shapes

def main():
    """Main function."""
    model_path = r'C:\Users\Davis\.insightface\models\buffalo_l\det_10g.onnx'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print("=" * 80)
    print("Spatial Scale Analysis")
    print("=" * 80)
    print(f"Model: {model_path}\n")
    
    model = onnx.load(model_path)
    
    # Method 1: Analyze downsampling operations
    print("\n" + "=" * 80)
    print("Method 1: Downsampling Operation Analysis")
    print("=" * 80)
    tensor_scale, scale_groups = analyze_downsampling_operations(model)
    
    # Method 2: Analyze Resize operations
    print("\n" + "=" * 80)
    print("Method 2: Resize Operation Analysis")
    print("=" * 80)
    resize_scales, tensor_shapes = identify_scales_from_resize_outputs(model)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Identified {len(scale_groups)} scale groups from downsampling analysis")
    print(f"Identified {len(resize_scales)} Resize operations")

if __name__ == "__main__":
    main()






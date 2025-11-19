"""
ONNX Model Diagnostic Script - Inspect graph structure and tensor shapes
"""

import onnx
import onnxruntime as ort
import numpy as np
import os
import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def inspect_model_graph(model_path: str, target_batch_size: int = 32):
    """Inspect the ONNX model graph structure and tensor shapes."""
    
    print(f"=" * 80)
    print(f"ONNX Model Graph Inspector")
    print(f"=" * 80)
    print(f"Model: {model_path}")
    print(f"Target batch size: {target_batch_size}")
    print()
    
    # Load model
    print("[LOAD] Loading model...")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Get input/output info
    print("\n" + "=" * 80)
    print("INPUT/OUTPUT INFORMATION")
    print("=" * 80)
    
    inputs = model.graph.input
    outputs = model.graph.output
    
    for i, inp in enumerate(inputs):
        print(f"\nInput {i}: {inp.name}")
        print(f"  Type: {inp.type}")
        if inp.type.tensor_type.shape:
            print(f"  Shape: {[dim.dim_value if dim.dim_value > 0 else (dim.dim_param if dim.dim_param else '?') for dim in inp.type.tensor_type.shape.dim]}")
    
    for i, out in enumerate(outputs):
        print(f"\nOutput {i}: {out.name}")
        print(f"  Type: {out.type}")
        if out.type.tensor_type.shape:
            print(f"  Shape: {[dim.dim_value if dim.dim_value > 0 else (dim.dim_param if dim.dim_param else '?') for dim in out.type.tensor_type.shape.dim]}")
    
    # Find Resize_144 node (current failing node)
    print("\n" + "=" * 80)
    print("SEARCHING FOR Resize_144 NODE (CURRENT ERROR)")
    print("=" * 80)
    
    resize_144_node = None
    resize_144_index = -1
    
    for i, node in enumerate(model.graph.node):
        if node.name == 'Resize_144':
            resize_144_node = node
            resize_144_index = i
            break
    
    if resize_144_node:
        print(f"[OK] Found Resize_144 at index {resize_144_index}")
        print(f"\nResize_144 Node Details:")
        print(f"  Name: {resize_144_node.name}")
        print(f"  Op Type: {resize_144_node.op_type}")
        print(f"  Inputs: {resize_144_node.input}")
        print(f"  Outputs: {resize_144_node.output}")
        
        # Detailed attribute inspection
        print(f"\n  Attributes (detailed):")
        for attr in resize_144_node.attribute:
            attr_name = attr.name
            attr_type = attr.type
            attr_value = None
            
            if attr_type == 1:  # FLOAT
                attr_value = attr.f
            elif attr_type == 2:  # INT
                attr_value = attr.i
            elif attr_type == 3:  # STRING
                attr_value = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s
            elif attr_type == 6:  # FLOATS
                attr_value = list(attr.floats)
            elif attr_type == 7:  # INTS
                attr_value = list(attr.ints)
            elif attr_type == 8:  # STRINGS
                attr_value = [s.decode('utf-8') if isinstance(s, bytes) else s for s in attr.strings]
            else:
                attr_value = f"<type {attr_type}>"
            
            print(f"    {attr_name}: {attr_value} (type={attr_type})")
        
        # Check for common Resize attributes that might cause DirectML issues
        resize_modes = []
        coord_transform_modes = []
        nearest_modes = []
        
        for attr in resize_144_node.attribute:
            if attr.name == 'mode':
                mode_val = attr.s.decode('utf-8') if attr.type == 3 else attr.i
                resize_modes.append(mode_val)
            elif attr.name == 'coordinate_transformation_mode':
                coord_val = attr.s.decode('utf-8') if attr.type == 3 else attr.i
                coord_transform_modes.append(coord_val)
            elif attr.name == 'nearest_mode':
                nearest_val = attr.s.decode('utf-8') if attr.type == 3 else attr.i
                nearest_modes.append(nearest_val)
        
        print(f"\n  Resize Configuration:")
        if resize_modes:
            print(f"    Mode: {resize_modes[0]} (may cause DirectML issues if not 'linear' or 'nearest')")
        if coord_transform_modes:
            print(f"    Coordinate transformation mode: {coord_transform_modes[0]} (may cause DirectML issues)")
        if nearest_modes:
            print(f"    Nearest mode: {nearest_modes[0]} (may cause DirectML issues)")
        
        # Find the input tensor for Resize_144
        resize_144_input = resize_144_node.input[0] if resize_144_node.input else None
        print(f"\n  Input tensor name: {resize_144_input}")
        
        # Also check all Resize inputs (data, roi, scales, sizes)
        print(f"  All inputs:")
        for i, inp_name in enumerate(resize_144_node.input):
            print(f"    Input {i}: {inp_name}")
    else:
        print("[WARN] Resize_144 node not found")
    
    # Find Transpose_221 node (previous error, now fixed)
    print("\n" + "=" * 80)
    print("SEARCHING FOR Transpose_221 NODE (PREVIOUS ERROR - NOW FIXED)")
    print("=" * 80)
    
    transpose_221_node = None
    transpose_221_index = -1
    
    for i, node in enumerate(model.graph.node):
        if node.name == 'Transpose_221':
            transpose_221_node = node
            transpose_221_index = i
            break
    
    if transpose_221_node is None:
        print("[WARN] Transpose_221 node not found by name, searching for Transpose nodes...")
        transpose_nodes = [(i, node) for i, node in enumerate(model.graph.node) if node.op_type == 'Transpose']
        print(f"Found {len(transpose_nodes)} Transpose nodes:")
        for idx, (i, node) in enumerate(transpose_nodes[:10]):  # Show first 10
            print(f"  [{idx}] Node {i}: name={node.name}, op_type={node.op_type}")
    else:
        print(f"[OK] Found Transpose_221 at index {transpose_221_index}")
    
    if transpose_221_node:
        print(f"\nTranspose_221 Node Details:")
        print(f"  Name: {transpose_221_node.name}")
        print(f"  Op Type: {transpose_221_node.op_type}")
        print(f"  Inputs: {transpose_221_node.input}")
        print(f"  Outputs: {transpose_221_node.output}")
        print(f"  Attributes: {[(attr.name, list(attr.ints) if attr.type == 7 else attr.i if attr.type == 2 else attr) for attr in transpose_221_node.attribute]}")
        
        # Find the input tensor for Transpose_221
        transpose_221_input = transpose_221_node.input[0] if transpose_221_node.input else None
        print(f"\n  Input tensor name: {transpose_221_input}")
        
        # Search for nodes that produce this input
        print(f"\n  Nodes producing '{transpose_221_input}':")
        for i, node in enumerate(model.graph.node):
            if transpose_221_input and transpose_221_input in node.output:
                print(f"    Node {i}: {node.name} ({node.op_type})")
                print(f"      Outputs: {node.output}")
    
    # Also find Relu_2 for comparison (previous error)
    print("\n" + "=" * 80)
    print("SEARCHING FOR Relu_2 NODE (PREVIOUS ERROR)")
    print("=" * 80)
    
    relu_2_node = None
    relu_2_index = -1
    
    for i, node in enumerate(model.graph.node):
        if node.name == 'Relu_2':
            relu_2_node = node
            relu_2_index = i
            break
    
    if relu_2_node:
        print(f"[OK] Found Relu_2 at index {relu_2_index}")
        relu_2_input = relu_2_node.input[0] if relu_2_node.input else None
        print(f"  Input tensor: {relu_2_input}")
    else:
        print("[WARN] Relu_2 node not found")
    
    # Inspect value_info for shape information
    print("\n" + "=" * 80)
    print("VALUE INFO (SHAPE INFORMATION)")
    print("=" * 80)
    
    # Get value_info from shape inference
    print("\nChecking value_info for intermediate tensors...")
    value_info_map = {}
    for vi in model.graph.value_info:
        value_info_map[vi.name] = vi
    
    # Check Resize_144 input shape if available
    if resize_144_node and resize_144_node.input:
        resize_144_input = resize_144_node.input[0] if resize_144_node.input else None
        print(f"\n[RESIZE_144] Checking input tensor '{resize_144_input}':")
        if resize_144_input and resize_144_input in value_info_map:
            vi = value_info_map[resize_144_input]
            print(f"[OK] Resize_144 input tensor '{resize_144_input}' shape info:")
            if vi.type.tensor_type.shape:
                shape = [dim.dim_value if dim.dim_value > 0 else (dim.dim_param if dim.dim_param else '?') for dim in vi.type.tensor_type.shape.dim]
                print(f"    Shape: {shape}")
                print(f"    Element type: {vi.type.tensor_type.elem_type}")
                print(f"    Number of dimensions: {len(shape)}")
        else:
            print(f"[WARN] Resize_144 input tensor '{resize_144_input}' not found in value_info")
            print(f"    This suggests shape inference may not have run properly or tensor is missing")
    
    # Check Transpose_221 input shape if available
    if transpose_221_node and transpose_221_node.input:
        transpose_221_input = transpose_221_node.input[0]
        print(f"\n[TRANSPOSE_221] Checking input tensor '{transpose_221_input}':")
        if transpose_221_input in value_info_map:
            vi = value_info_map[transpose_221_input]
            print(f"[OK] Transpose_221 input tensor '{transpose_221_input}' shape info:")
            if vi.type.tensor_type.shape:
                shape = [dim.dim_value if dim.dim_value > 0 else (dim.dim_param if dim.dim_param else '?') for dim in vi.type.tensor_type.shape.dim]
                print(f"    Shape: {shape}")
                print(f"    Element type: {vi.type.tensor_type.elem_type}")
                print(f"    Number of dimensions: {len(shape)}")
                # Check if batch dimension is correct
                if len(shape) > 0:
                    batch_dim = shape[0]
                    if batch_dim != target_batch_size and batch_dim != '?' and batch_dim != 'batch':
                        print(f"    [WARN] Batch dimension is {batch_dim}, expected {target_batch_size}")
        else:
            print(f"[WARN] Transpose_221 input tensor '{transpose_221_input}' not found in value_info")
            print(f"    This suggests shape inference may not have run properly or tensor is missing")
    
    # Check Relu_2 input shape if available (for comparison)
    if relu_2_node and relu_2_node.input:
        relu_2_input = relu_2_node.input[0]
        if relu_2_input in value_info_map:
            vi = value_info_map[relu_2_input]
            print(f"\n[OK] Relu_2 input tensor '{relu_2_input}' shape info:")
            if vi.type.tensor_type.shape:
                shape = [dim.dim_value if dim.dim_value > 0 else (dim.dim_param if dim.dim_param else '?') for dim in vi.type.tensor_type.shape.dim]
                print(f"    Shape: {shape}")
                print(f"    Element type: {vi.type.tensor_type.elem_type}")
    
    # Count tensors with incorrect batch dimensions
    print(f"\n[DIAG] Analyzing all 4D activation tensors for batch dimension issues...")
    incorrect_batch_tensors = []
    for vi in model.graph.value_info:
        if vi.type.tensor_type.shape and len(vi.type.tensor_type.shape.dim) == 4:
            batch_dim = vi.type.tensor_type.shape.dim[0]
            if batch_dim.HasField('dim_value'):
                if batch_dim.dim_value == 1:
                    incorrect_batch_tensors.append((vi.name, batch_dim.dim_value))
            elif batch_dim.HasField('dim_param'):
                # Dynamic batch is OK
                pass
    
    if incorrect_batch_tensors:
        print(f"[WARN] Found {len(incorrect_batch_tensors)} 4D tensors with batch=1:")
        for name, batch_val in incorrect_batch_tensors[:10]:
            print(f"    {name}: batch={batch_val}")
        if len(incorrect_batch_tensors) > 10:
            print(f"    ... and {len(incorrect_batch_tensors) - 10} more")
    else:
        print(f"[OK] No 4D tensors found with batch=1")
    
    # List nodes around the failing nodes to understand graph structure
    print("\n" + "=" * 80)
    print("GRAPH STRUCTURE AROUND FAILING NODES")
    print("=" * 80)
    
    # Show nodes around Transpose_221
    if transpose_221_index >= 0:
        print(f"\nNodes around Transpose_221 (index {transpose_221_index}):")
        start_idx = max(0, transpose_221_index - 5)
        end_idx = min(len(model.graph.node), transpose_221_index + 6)
        for i in range(start_idx, end_idx):
            node = model.graph.node[i]
            marker = ""
            if i == transpose_221_index:
                marker = " [*** CURRENT ERROR NODE ***]"
            print(f"  [{i}] {node.name} ({node.op_type}){marker}")
            print(f"      Inputs: {node.input}")
            print(f"      Outputs: {node.output}")
    
    # Show nodes around Relu_2 for comparison
    if relu_2_index >= 0:
        print(f"\nNodes around Relu_2 (index {relu_2_index}):")
        start_idx = max(0, relu_2_index - 5)
        end_idx = min(len(model.graph.node), relu_2_index + 6)
        for i in range(start_idx, end_idx):
            node = model.graph.node[i]
            marker = ""
            if i == relu_2_index:
                marker = " [PREVIOUS ERROR NODE]"
            print(f"  [{i}] {node.name} ({node.op_type}){marker}")
            print(f"      Inputs: {node.input}")
            print(f"      Outputs: {node.output}")
    
    # Also show first 10 nodes
    print(f"\nFirst 10 nodes in graph:")
    for i, node in enumerate(model.graph.node[:10]):
        print(f"  [{i}] {node.name} ({node.op_type})")
        print(f"      Inputs: {node.input}")
        print(f"      Outputs: {node.output}")
    
    # Try to run shape inference with test input
    print("\n" + "=" * 80)
    print("TESTING WITH ONNX RUNTIME")
    print("=" * 80)
    
    try:
        # Create session to inspect
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Error only
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        
        print(f"[OK] Session created")
        print(f"   Providers: {session.get_providers()}")
        
        # Get input info
        input_info = session.get_inputs()[0]
        print(f"\n   Input: {input_info.name}")
        print(f"   Expected shape: {input_info.shape}")
        print(f"   Type: {input_info.type}")
        
        # Create test input
        # Handle dynamic spatial dimensions
        test_shape = (target_batch_size, 3, 640, 640)
        if isinstance(input_info.shape[2], int) and isinstance(input_info.shape[3], int):
            test_shape = tuple(input_info.shape)
        
        print(f"\n   Creating test input with shape: {test_shape}")
        test_input = np.random.randn(*test_shape).astype(np.float32)
        print(f"   Test input shape: {test_input.shape}, dtype: {test_input.dtype}")
        
        # Try to get output shapes (this might fail if DirectML has issues)
        print(f"\n   Attempting inference...")
        try:
            outputs = session.run(None, {input_info.name: test_input})
            print(f"   [OK] Inference succeeded!")
            for i, out in enumerate(outputs):
                print(f"   Output {i}: shape={out.shape}, dtype={out.dtype}")
        except Exception as e:
            print(f"   [ERROR] Inference failed: {type(e).__name__}: {e}")
            print(f"   This confirms the DirectML error we're seeing")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"[ERROR] Failed to create session or test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Find the modified model
    home = os.path.expanduser("~/.insightface")
    model_dir = os.path.join(home, "models", "buffalo_l")
    model_path = os.path.join(model_dir, "det_10g_modified.onnx")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    
    # Get batch size from config or default
    batch_size = 32  # Default, can be changed
    if len(sys.argv) > 1:
        batch_size = int(sys.argv[1])
    
    inspect_model_graph(model_path, batch_size)


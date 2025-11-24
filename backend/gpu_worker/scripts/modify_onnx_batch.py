"""
ONNX Model Batch Dimension Modification Script

Modifies InsightFace detection model to support dynamic batch dimensions,
enabling true batch processing with ONNX Runtime.

This script:
1. Locates the InsightFace detection model (det_10g.onnx or detection.onnx)
2. Modifies the model to support dynamic batch dimensions
3. Runs shape inference to propagate batch dimension through the graph
4. Saves a new _dynamic.onnx file (original remains unchanged)
5. Validates the modified model before saving
"""

import onnx
import onnx.checker
from onnx import shape_inference
import os
import sys
import shutil
import io
from pathlib import Path
from collections import defaultdict

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def validate_model(model_path: str) -> bool:
    """Validate that an ONNX model is valid."""
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        return True
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False


def find_model_path(batch_size: int = None) -> tuple[str, str]:
    """
    Find the InsightFace detection model path.
    
    Args:
        batch_size: If provided, use fixed batch size. If None, use dynamic batch.
                   Note: Output file is always named with "_modified" suffix.
    
    Returns:
        Tuple of (model_path, output_path) or (None, None) if not found
    """
    home = os.path.expanduser("~/.insightface")
    model_dir = os.path.join(home, "models", "buffalo_l")
    
    # Always use "_modified" suffix for output (regardless of batch size)
    # This allows changing batch sizes without hardcoding model names
    suffix = "_modified"
    
    # Try primary filename first
    primary_path = os.path.join(model_dir, "det_10g.onnx")
    if os.path.exists(primary_path):
        base, ext = os.path.splitext(primary_path)
        output_path = f"{base}{suffix}{ext}"
        return primary_path, output_path
    
    # Try fallback filename
    fallback_path = os.path.join(model_dir, "detection.onnx")
    if os.path.exists(fallback_path):
        base, ext = os.path.splitext(fallback_path)
        output_path = f"{base}{suffix}{ext}"
        return fallback_path, output_path
    
    return None, None


def identify_scales_from_shape_inference(model: onnx.ModelProto) -> dict:
    """
    Identify scales by analyzing spatial dimensions from shape inference results.
    
    This method groups tensors by their spatial dimensions (H, W) to identify
    which tensors are at the same scale. Tensors with the same H/W parameters
    or values are grouped together.
    """
    print("🔍 Identifying scales from shape inference results...")
    
    # Model should already have shape inference run, so we can analyze the shapes directly
    
    # Build tensor shape map
    tensor_shapes = {}
    for vi in model.graph.value_info:
        if vi.type.tensor_type.shape and len(vi.type.tensor_type.shape.dim) == 4:
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            if len(shape) == 4:
                tensor_shapes[vi.name] = shape
    
    # Group tensors by their spatial dimensions (H and W)
    # Tensors with the same H/W should be at the same scale
    spatial_groups = {}  # (h_dim, w_dim) -> list of tensor names
    
    for tensor_name, shape in tensor_shapes.items():
        if len(shape) == 4:
            batch_dim = shape[0]
            channel_dim = shape[1]
            h_dim = shape[2]
            w_dim = shape[3]
            
            # Create a key based on spatial dimensions
            # Use dim_param if available, otherwise use dim_value or '?'
            spatial_key = (h_dim, w_dim)
            
            if spatial_key not in spatial_groups:
                spatial_groups[spatial_key] = []
            spatial_groups[spatial_key].append(tensor_name)
    
    print(f"   Found {len(spatial_groups)} distinct spatial dimension groups")
    
    # Assign scale IDs to each spatial group
    tensor_to_scale = {}
    scale_id = 0
    scale_groups = {}
    
    for spatial_key, tensor_list in sorted(spatial_groups.items()):
        h_dim, w_dim = spatial_key
        scale_groups[scale_id] = set(tensor_list)
        for tensor_name in tensor_list:
            tensor_to_scale[tensor_name] = scale_id
        
        print(f"   Scale {scale_id}: H={h_dim}, W={w_dim} ({len(tensor_list)} tensors)")
        scale_id += 1
    
    return tensor_to_scale, scale_groups


def identify_scale_groups(model: onnx.ModelProto) -> dict:
    """
    Identify scale groups in multi-scale architecture by analyzing:
    1. Resize nodes (explicit scale changes)
    2. Downsampling operations (Conv with stride > 1, MaxPool, etc.)
    3. Spatial dimension relationships
    
    Returns:
        Tuple of (tensor_to_scale dict, scale_groups dict)
    """
    print("🔍 Identifying scale groups from network structure...")
    
    # Build a map of tensor names to their scale groups
    tensor_to_scale = {}
    
    # Build dependency graphs
    tensor_consumers = {}  # tensor_name -> list of nodes that consume it
    tensor_producers = {}  # tensor_name -> node that produces it
    
    for node in model.graph.node:
        for output in node.output:
            tensor_producers[output] = node
        for input_name in node.input:
            if input_name not in tensor_consumers:
                tensor_consumers[input_name] = []
            tensor_consumers[input_name].append(node)
    
    # Step 1: Identify scale anchors (Resize nodes and major downsampling operations)
    scale_anchors = []  # List of (scale_id, anchor_tensor, anchor_node)
    scale_id = 0
    
    # Find Resize nodes (explicit scale boundaries)
    resize_nodes = [node for node in model.graph.node if node.op_type == 'Resize']
    print(f"   Found {len(resize_nodes)} Resize nodes")
    
    # Create scales for Resize outputs (each Resize output is a new scale)
    resize_output_to_scale = {}  # resize_output_tensor -> scale_id
    for resize_node in resize_nodes:
        if resize_node.output:
            anchor_tensor = resize_node.output[0]
            resize_output_to_scale[anchor_tensor] = scale_id
            scale_anchors.append((scale_id, anchor_tensor, resize_node))
            print(f"   Scale {scale_id}: Resize '{resize_node.name}' -> output '{anchor_tensor}'")
            scale_id += 1
    
    # Step 2: Identify downsampling operations that create new scales
    # (Resize inputs will be handled later based on depth and connections)
    # Look for Conv/MaxPool/AveragePool with stride > 1
    input_tensor = model.graph.input[0].name
    visited_nodes = set()
    
    def get_stride(node):
        """Get stride from Conv/Pool node."""
        for attr in node.attribute:
            if attr.name == 'strides':
                if attr.type == 7:  # INTS
                    strides = list(attr.ints)
                    if len(strides) >= 2:
                        return max(strides[0], strides[1])
        return 1
    
    # Track scale depth from input (number of downsampling operations)
    tensor_scale_depth = {input_tensor: 0}  # tensor_name -> downsampling depth
    
    # Process nodes in topological order to track scale depth
    changed = True
    iterations = 0
    max_iterations = 200
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        for node in model.graph.node:
            # Get input scales
            input_depths = []
            for inp in node.input:
                if inp in tensor_scale_depth:
                    input_depths.append(tensor_scale_depth[inp])
            
            if not input_depths:
                continue
            
            base_depth = max(input_depths) if input_depths else 0
            output_depth = base_depth
            
            # Check if this is a downsampling operation
            if node.op_type in ['Conv', 'MaxPool', 'AveragePool']:
                stride = get_stride(node)
                if stride > 1:
                    output_depth = base_depth + 1
                    # This creates a potential new scale
                    if node.output and node.output[0] not in tensor_scale_depth:
                        # Check if this depth already has a scale anchor
                        existing_scale_for_depth = None
                        for sid, anchor_tensor, anchor_node in scale_anchors:
                            if anchor_tensor in tensor_scale_depth:
                                if tensor_scale_depth[anchor_tensor] == output_depth:
                                    existing_scale_for_depth = sid
                                    break
                        
                        if existing_scale_for_depth is None:
                            # Create a new scale anchor for this downsampling level
                            scale_anchors.append((scale_id, node.output[0], node))
                            print(f"   Scale {scale_id}: Downsampling '{node.op_type}' '{node.name}' (stride={stride}) -> output '{node.output[0]}'")
                            scale_id += 1
            
            # Update output depths
            for output in node.output:
                if output not in tensor_scale_depth or tensor_scale_depth[output] != output_depth:
                    tensor_scale_depth[output] = output_depth
                    changed = True
    
    print(f"   Identified {len(scale_anchors)} scale anchors (Resize + downsampling)")
    print(f"   Tracked scale depth for {len(tensor_scale_depth)} tensors")
    
    # Step 3: Assign scale IDs to tensors based on scale anchors and depth
    scale_groups = {}  # scale_id -> set of tensor names
    
    # Initialize scale groups with anchors
    # For Resize nodes, the input and output are at DIFFERENT scales
    # So we need to handle them separately
    resize_input_scales = {}  # resize_input -> scale_id for input scale
    
    for scale_id, anchor_tensor, anchor_node in scale_anchors:
        scale_groups[scale_id] = {anchor_tensor}
        tensor_to_scale[anchor_tensor] = scale_id
        
        # For Resize nodes, the OUTPUT is at this scale
        # The INPUT should be at a different scale (we'll handle it separately)
        if anchor_node.op_type == 'Resize' and anchor_node.input:
            resize_input = anchor_node.input[0]
            # Don't assign input to output scale - they're different!
            # We'll create a separate scale for the input or assign it to an existing scale
    
    # Step 4: Assign tensors to scales by tracing connections from each anchor
    # Process each scale anchor independently to build scale groups
    for scale_id, anchor_tensor, anchor_node in scale_anchors:
        if scale_id not in scale_groups:
            scale_groups[scale_id] = set()
        
        visited = set()
        queue = [anchor_tensor]
        visited.add(anchor_tensor)
        
        # Forward traversal: from anchor to downstream tensors
        # Stop at scale boundaries (downsampling operations, Resize nodes)
        while queue:
            current_tensor = queue.pop(0)
            scale_groups[scale_id].add(current_tensor)
            # Only assign if not already assigned to a different scale
            if current_tensor not in tensor_to_scale:
                tensor_to_scale[current_tensor] = scale_id
            elif tensor_to_scale[current_tensor] != scale_id:
                # Conflict: tensor belongs to multiple scales
                # Keep the first assignment (or we could prefer Resize-based scales)
                continue
            
            if current_tensor in tensor_consumers:
                for consumer_node in tensor_consumers[current_tensor]:
                    # Check if this is a scale boundary
                    is_scale_boundary = False
                    
                    # Resize operations are scale boundaries
                    if consumer_node.op_type == 'Resize':
                        is_scale_boundary = True
                    # Downsampling operations are scale boundaries
                    elif consumer_node.op_type in ['Conv', 'MaxPool', 'AveragePool']:
                        stride = get_stride(consumer_node)
                        if stride > 1:
                            is_scale_boundary = True
                            # Check if this downsampling is already an anchor
                            # If so, it's a boundary; if not, we might want to cross it
                            is_anchor = False
                            consumer_node_name = consumer_node.name if hasattr(consumer_node, 'name') else str(consumer_node)
                            for _, other_anchor_tensor, other_anchor_node in scale_anchors:
                                other_node_name = other_anchor_node.name if hasattr(other_anchor_node, 'name') else str(other_anchor_node)
                                if other_node_name == consumer_node_name:
                                    is_anchor = True
                                    break
                            # Downsampling operations are always scale boundaries
                            is_scale_boundary = True
                    
                    if not is_scale_boundary:
                        for output_tensor in consumer_node.output:
                            if output_tensor and output_tensor not in visited:
                                # Only process 4D tensors
                                is_4d = False
                                for vi in model.graph.value_info:
                                    if vi.name == output_tensor and vi.type.tensor_type.shape:
                                        if len(vi.type.tensor_type.shape.dim) == 4:
                                            is_4d = True
                                            break
                                
                                if not is_4d:
                                    for inp in model.graph.input:
                                        if inp.name == output_tensor and inp.type.tensor_type.shape:
                                            if len(inp.type.tensor_type.shape.dim) == 4:
                                                is_4d = True
                                                break
                                
                                if is_4d:
                                    visited.add(output_tensor)
                                    queue.append(output_tensor)
        
        # Backward traversal: from anchor to upstream tensors
        # Only for non-Resize anchors (Resize inputs are handled separately)
        if anchor_node.op_type != 'Resize':
            visited_backward = set()
            queue_backward = [anchor_tensor]
            visited_backward.add(anchor_tensor)
            
            while queue_backward:
                current_tensor = queue_backward.pop(0)
                scale_groups[scale_id].add(current_tensor)
                if current_tensor not in tensor_to_scale:
                    tensor_to_scale[current_tensor] = scale_id
                elif tensor_to_scale[current_tensor] != scale_id:
                    continue
                
                if current_tensor in tensor_producers:
                    producer_node = tensor_producers[current_tensor]
                    # Check if this is a scale boundary
                    is_scale_boundary = False
                    
                    if producer_node.op_type == 'Resize':
                        is_scale_boundary = True
                    elif producer_node.op_type in ['Conv', 'MaxPool', 'AveragePool']:
                        stride = get_stride(producer_node)
                        if stride > 1:
                            is_scale_boundary = True
                    
                    if not is_scale_boundary:
                        for input_tensor in producer_node.input:
                            if input_tensor and input_tensor not in visited_backward:
                                # Only process 4D tensors
                                is_4d = False
                                for vi in model.graph.value_info:
                                    if vi.name == input_tensor and vi.type.tensor_type.shape:
                                        if len(vi.type.tensor_type.shape.dim) == 4:
                                            is_4d = True
                                            break
                                
                                if not is_4d:
                                    for inp in model.graph.input:
                                        if inp.name == input_tensor and inp.type.tensor_type.shape:
                                            if len(inp.type.tensor_type.shape.dim) == 4:
                                                is_4d = True
                                                break
                                
                                if is_4d:
                                    visited_backward.add(input_tensor)
                                    queue_backward.append(input_tensor)
        
        # For Resize nodes, also handle the input separately
        # The input should be assigned to a scale based on its depth and connections
        if anchor_node.op_type == 'Resize' and anchor_node.input:
            resize_input = anchor_node.input[0]
            # Find or create a scale for the input
            # Check if input depth matches any existing scale
            input_depth = tensor_scale_depth.get(resize_input, -1)
            
            # Find scales at the same depth
            input_scale_id = None
            for other_scale_id, other_anchor_tensor, other_anchor_node in scale_anchors:
                if other_anchor_tensor in tensor_scale_depth:
                    other_depth = tensor_scale_depth[other_anchor_tensor]
                    if other_depth == input_depth and other_scale_id != scale_id:
                        input_scale_id = other_scale_id
                        break
            
            # If no matching scale found, use the input's current assignment or create new logic
            # For now, trace backward from input to find its scale
            if resize_input not in tensor_to_scale:
                # Trace backward to find what scale this input belongs to
                visited_input = set([resize_input])
                queue_input = [resize_input]
                
                while queue_input:
                    current_tensor = queue_input.pop(0)
                    # Assign to a scale based on depth
                    tensor_depth = tensor_scale_depth.get(current_tensor, -1)
                    # Find a scale at this depth (prefer non-Resize scales)
                    assigned = False
                    for other_scale_id, other_anchor_tensor, other_anchor_node in scale_anchors:
                        if other_anchor_tensor in tensor_scale_depth:
                            other_depth = tensor_scale_depth[other_anchor_tensor]
                            if other_depth == tensor_depth and other_anchor_node.op_type != 'Resize':
                                if current_tensor not in tensor_to_scale:
                                    tensor_to_scale[current_tensor] = other_scale_id
                                    if other_scale_id not in scale_groups:
                                        scale_groups[other_scale_id] = set()
                                    scale_groups[other_scale_id].add(current_tensor)
                                    assigned = True
                                    break
                    
                    if not assigned and current_tensor in tensor_producers:
                        producer_node = tensor_producers[current_tensor]
                        for input_tensor in producer_node.input:
                            if input_tensor and input_tensor not in visited_input:
                                is_4d = False
                                for vi in model.graph.value_info:
                                    if vi.name == input_tensor and vi.type.tensor_type.shape:
                                        if len(vi.type.tensor_type.shape.dim) == 4:
                                            is_4d = True
                                            break
                                if is_4d:
                                    visited_input.add(input_tensor)
                                    queue_input.append(input_tensor)
        
        print(f"   Scale {scale_id}: {len(scale_groups[scale_id])} tensors")
    
    return tensor_to_scale, scale_groups, tensor_scale_depth


def assign_unique_spatial_parameters(model: onnx.ModelProto, tensor_to_scale: dict, fixed_batch_size: int, tensor_scale_depth: dict = None) -> int:
    """
    Assign unique spatial dimension parameters to different scales.
    
    Args:
        model: ONNX model to modify
        tensor_to_scale: Dictionary mapping tensor names to scale IDs
        fixed_batch_size: Fixed batch size (for batch dimension)
        tensor_scale_depth: Optional dictionary mapping tensor names to downsampling depth
    
    Returns:
        Number of tensors updated
    """
    print(f"🔧 Assigning unique spatial dimension parameters to {len(set(tensor_to_scale.values()))} scale groups...")
    
    # Create parameter names for each scale
    # Format: 'H0', 'W0' for scale 0, 'H1', 'W1' for scale 1, etc.
    max_scale = max(tensor_to_scale.values()) if tensor_to_scale else 0
    scale_params = {}
    for scale_id in range(max_scale + 1):
        scale_params[scale_id] = (f'H{scale_id}', f'W{scale_id}')
    
    # If we have depth information, also use it to refine assignments
    # Tensors at different depths should have different parameters
    if tensor_scale_depth:
        # Create depth-based parameter mapping
        depth_to_param = {}  # depth -> (H_param, W_param)
        depth_param_id = 0
        for tensor_name, scale_id in tensor_to_scale.items():
            depth = tensor_scale_depth.get(tensor_name, -1)
            if depth not in depth_to_param:
                depth_to_param[depth] = (f'H{depth_param_id}', f'W{depth_param_id}')
                depth_param_id += 1
        print(f"   Using depth-based parameters for {len(depth_to_param)} depths")
    
    # Get weight/bias names (should not be modified)
    weight_bias_names = {init.name for init in model.graph.initializer}
    
    # Get shape parameter tensors (should not be modified)
    shape_parameter_tensors = set()
    for node in model.graph.node:
        if node.op_type == 'Shape':
            if node.output:
                shape_parameter_tensors.add(node.output[0])
        elif node.op_type in ['Gather', 'Scatter', 'GatherND', 'GatherElements']:
            if len(node.input) > 1:
                shape_parameter_tensors.add(node.input[1])
        elif node.op_type == 'Resize':
            # Resize sizes/scales inputs are shape parameters
            if len(node.input) >= 3:
                for inp in node.input[2:]:
                    if inp and inp not in weight_bias_names:
                        shape_parameter_tensors.add(inp)
    
    tensors_updated = 0
    
    # Update all 4D tensors with unique spatial parameters based on their scale
    all_tensors = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    
    for tensor in all_tensors:
        # Skip if not a 4D tensor
        if not tensor.type.tensor_type.shape or len(tensor.type.tensor_type.shape.dim) != 4:
            continue
        
        # Skip if it's a weight/bias or shape parameter
        if tensor.name in weight_bias_names or tensor.name in shape_parameter_tensors:
            continue
        
        # Get scale ID for this tensor
        scale_id = tensor_to_scale.get(tensor.name)
        
        # Prioritize depth-based parameters if available (more accurate for spatial dimensions)
        if tensor_scale_depth and tensor.name in tensor_scale_depth:
            depth = tensor_scale_depth[tensor.name]
            if depth in depth_to_param:
                h_param, w_param = depth_to_param[depth]
            elif scale_id is not None:
                h_param, w_param = scale_params[scale_id]
            else:
                continue
        elif scale_id is not None:
            h_param, w_param = scale_params[scale_id]
        else:
            # Tensor doesn't belong to any identified scale - use default '?' parameters
            continue
        
        # Update spatial dimensions (indices 2 and 3 for 4D tensors)
        dims = tensor.type.tensor_type.shape.dim
        if len(dims) >= 4:
            # Update H dimension (index 2)
            h_dim = dims[2]
            if h_dim.HasField('dim_param'):
                if h_dim.dim_param == '?' or h_dim.dim_param.startswith('unk__'):
                    h_dim.dim_param = h_param
                    tensors_updated += 1
            elif not h_dim.HasField('dim_value'):  # Unknown dimension
                h_dim.dim_param = h_param
                tensors_updated += 1
            
            # Update W dimension (index 3)
            w_dim = dims[3]
            if w_dim.HasField('dim_param'):
                if w_dim.dim_param == '?' or w_dim.dim_param.startswith('unk__'):
                    w_dim.dim_param = w_param
                    tensors_updated += 1
            elif not w_dim.HasField('dim_value'):  # Unknown dimension
                w_dim.dim_param = w_param
                tensors_updated += 1
    
    print(f"✅ Updated {tensors_updated} tensors with unique spatial parameters")
    
    # Print scale parameter mapping
    for scale_id in sorted(scale_params.keys()):
        h_param, w_param = scale_params[scale_id]
        scale_tensors = [name for name, sid in tensor_to_scale.items() if sid == scale_id]
        print(f"   Scale {scale_id}: {h_param}, {w_param} ({len(scale_tensors)} tensors)")
    
    return tensors_updated


def modify_model_for_batching(model_path: str, output_path: str = None, backup: bool = True, fixed_batch_size: int = None) -> bool:
    """
    Modify ONNX model to support batch processing.
    
    Args:
        model_path: Path to the original ONNX model (will NOT be modified)
        output_path: Path to save the modified model (default: adds '_batch512' suffix)
        backup: If output_path exists, create a backup first
        fixed_batch_size: If provided, use fixed batch size (e.g., 512). If None, use dynamic batch
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Validate original model first
    print(f"🔍 Validating original model...")
    if not validate_model(model_path):
        print(f"❌ Original model is invalid or corrupted!")
        return False
    print(f"✅ Original model is valid")
    
    if output_path is None:
        base, ext = os.path.splitext(model_path)
        output_path = f"{base}_modified{ext}"
    
    # If output exists, handle it
    if os.path.exists(output_path):
        if backup:
            backup_path = f"{output_path}.backup"
            print(f"📦 Output file exists, creating backup: {backup_path}")
            shutil.copy2(output_path, backup_path)
        else:
            response = input(f"⚠️  Output file exists: {output_path}\nOverwrite? (y/n): ")
            if response.lower() != 'y':
                print("❌ Cancelled")
                return False
    
    print(f"📥 Loading model: {model_path}")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Get input tensor
    if len(model.graph.input) == 0:
        print("❌ Model has no inputs!")
        return False
    
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    print(f"📝 Input name: {input_name}")
    
    # Check current batch dimension
    shape = input_tensor.type.tensor_type.shape
    if len(shape.dim) == 0:
        print("❌ Input has no shape information!")
        return False
    
    batch_dim = shape.dim[0]
    
    # Check if already dynamic
    if batch_dim.HasField('dim_param'):
        param_name = batch_dim.dim_param
        print(f"✅ Model already has dynamic batch dimension: '{param_name}'")
        # Still create the output file for consistency
        if not os.path.exists(output_path):
            print(f"💾 Copying model to: {output_path}")
            shutil.copy2(model_path, output_path)
        return True
    
    batch_value = batch_dim.dim_value if batch_dim.HasField('dim_value') else None
    print(f"📊 Current batch dimension: {batch_value} (fixed)")
    
    # Make batch dimension fixed or dynamic
    # CRITICAL: Keep spatial dimensions dynamic ('?') - DO NOT fix them to concrete values
    # The SCRFD model is designed for dynamic spatial dimensions. Letterbox preprocessing
    # ensures consistent 640x640 at runtime, but the model graph expects dynamic spatial dims.
    # Fixing spatial dimensions causes DirectML issues with Resize operations.
    if fixed_batch_size is not None:
        print(f"🔧 Setting batch dimension to fixed size: {fixed_batch_size}...")
        print(f"   Keeping spatial dimensions dynamic ('?') to match original model design")
        batch_dim.ClearField('dim_param')
        batch_dim.dim_value = fixed_batch_size
        batch_type = f"fixed={fixed_batch_size}"
        # Note: Spatial dimensions remain dynamic ('?') - this is intentional and correct
    else:
        print("🔧 Making batch dimension dynamic...")
        batch_dim.ClearField('dim_value')
        batch_dim.dim_param = 'batch'
        batch_type = "dynamic"
    
    # Update outputs
    outputs_updated = 0
    for output in model.graph.output:
        output_shape = output.type.tensor_type.shape
        if len(output_shape.dim) > 0:
            first_dim = output_shape.dim[0]
            if fixed_batch_size is not None:
                # Set fixed batch size for outputs
                if first_dim.HasField('dim_value') and first_dim.dim_value > 0:
                    first_dim.dim_value = fixed_batch_size
                    outputs_updated += 1
                elif first_dim.HasField('dim_param'):
                    first_dim.ClearField('dim_param')
                    first_dim.dim_value = fixed_batch_size
                    outputs_updated += 1
            else:
                # Set dynamic batch for outputs
                if first_dim.HasField('dim_value') and first_dim.dim_value > 0:
                    first_dim.ClearField('dim_value')
                    first_dim.dim_param = 'batch'
                    outputs_updated += 1
    
    if outputs_updated > 0:
        print(f"✅ Updated {outputs_updated} output(s) batch dimension")
    
    # CRITICAL: Run shape inference to propagate batch dimension through the graph
    # This ensures all intermediate tensors have correct batch shapes
    if fixed_batch_size is None:
        print("🔧 Running shape inference to propagate dynamic batch dimension...")
    else:
        print(f"🔧 Running shape inference to propagate fixed batch dimension ({fixed_batch_size})...")
    try:
        # Shape inference will propagate the batch dimension through all intermediate tensors
        inferred_model = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
        model = inferred_model
        print("✅ Shape inference completed")
        
        # Fix 'unk__' dimensions after first shape inference
        # CRITICAL: Replace 'unk__' with '?' (dynamic) for spatial dimensions
        # We keep spatial dimensions dynamic to match the original model design and avoid DirectML issues
        if fixed_batch_size is not None:
            print(f"🔧 Fixing 'unk__' dimensions after first shape inference...")
            print(f"   Replacing 'unk__' with '?' (dynamic) for spatial dimensions")
            unk_fixed = 0
            # Check all tensors (inputs, outputs, value_info)
            all_tensors = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
            for tensor in all_tensors:
                if tensor.type.tensor_type.shape:
                    dims = tensor.type.tensor_type.shape.dim
                    for dim_index, dim in enumerate(dims):
                        if dim.HasField('dim_param'):
                            dim_param = dim.dim_param
                            if dim_param.startswith('unk__'):
                                # Replace 'unk__' with '?' (dynamic) for all dimensions
                                # This matches the original model design where spatial dimensions are dynamic
                                dim.ClearField('dim_param')
                                dim.dim_param = '?'  # Keep dynamic - this is correct for SCRFD models
                                unk_fixed += 1
                                if unk_fixed <= 5:
                                    print(f"   Fixed '{tensor.name}' dim[{dim_index}]: '{dim_param}' -> '?' (dynamic)")
            if unk_fixed > 0:
                print(f"✅ Fixed {unk_fixed} 'unk__' dimensions to '?' (dynamic)")
            else:
                print(f"✅ No 'unk__' dimensions found (good!)")
        
        # MANUAL FIX: If fixed batch size, manually update all intermediate tensor batch dimensions
        # Shape inference with dynamic spatial dimensions sometimes fails to update batch dims
        # We need to fix activation tensors (outputs of Conv, Relu, etc.) but not weights/biases
        if fixed_batch_size is not None:
            print(f"🔧 Manually fixing intermediate tensor batch dimensions to {fixed_batch_size}...")
            tensors_fixed = 0
            
            # Get list of weight/bias tensor names (these shouldn't have batch dimension)
            weight_bias_names = set()
            for init in model.graph.initializer:
                weight_bias_names.add(init.name)
            
            # Get list of node outputs that are activations (not weights)
            activation_tensors = set()
            for node in model.graph.node:
                for output in node.output:
                    if output not in weight_bias_names:
                        activation_tensors.add(output)
            
            # Identify shape parameter tensors (used as sizes, shapes, etc.)
            # These should NOT have their batch dimensions changed
            shape_parameter_tensors = set()
            for node in model.graph.node:
                op_type = node.op_type
                # Resize uses 'sizes' or 'scales' as shape parameters (typically 2nd or 3rd input)
                if op_type == 'Resize':
                    # Resize has inputs: [data, roi, scales, sizes] - sizes/scales are shape params
                    # We need to identify which input is the sizes/scales
                    if len(node.input) >= 2:
                        # Check if any input after data is a shape parameter
                        for i, inp in enumerate(node.input[1:], start=1):
                            if inp and inp not in weight_bias_names:
                                shape_parameter_tensors.add(inp)
                # Shape operation outputs are shape parameters
                elif op_type == 'Shape':
                    if node.output:
                        shape_parameter_tensors.add(node.output[0])
                # Gather/Scatter operations use indices (typically 2nd input)
                elif op_type in ['Gather', 'Scatter', 'GatherND', 'GatherElements']:
                    if len(node.input) > 1:
                        shape_parameter_tensors.add(node.input[1])
            
            # Also identify tensors that are likely shape parameters by their dimensions
            # Shape parameters are typically 1D tensors with small sizes (4 elements for 4D shape)
            for vi in model.graph.value_info:
                if vi.type.tensor_type.shape:
                    dims = vi.type.tensor_type.shape.dim
                    # Skip 1D tensors with small sizes (likely shape parameters)
                    if len(dims) == 1:
                        dim_size = dims[0].dim_value if dims[0].HasField('dim_value') else None
                        # If it's a 1D tensor with <= 4 elements, it's likely a shape parameter
                        if dim_size is not None and dim_size <= 4:
                            shape_parameter_tensors.add(vi.name)
            
            # Fix value_info (intermediate tensors) that are activations
            for vi in model.graph.value_info:
                # Skip if this is a weight/bias tensor
                if vi.name in weight_bias_names:
                    continue
                
                # Skip if this is a shape parameter tensor
                if vi.name in shape_parameter_tensors:
                    continue
                
                # Only fix if it's an activation tensor (output of a node)
                if vi.name not in activation_tensors:
                    continue
                
                # Only fix 4D tensors (N, C, H, W) - image-like activation tensors
                # Skip 1D, 2D, 3D tensors that might be shape parameters or other non-image tensors
                if not vi.type.tensor_type.shape or len(vi.type.tensor_type.shape.dim) != 4:
                    continue
                
                if vi.type.tensor_type.shape and len(vi.type.tensor_type.shape.dim) > 0:
                    batch_dim = vi.type.tensor_type.shape.dim[0]
                    needs_fix = False
                    old_value = None
                    
                    # Only fix if the tensor has batch=1 (from original model)
                    # This is the safe case - we know it should be batch=32 for fixed batch model
                    # Don't fix tensors with other batch sizes as they might be correct
                    if batch_dim.HasField('dim_value'):
                        # Only fix if batch dimension is 1 (original model had batch=1)
                        if batch_dim.dim_value == 1:
                            old_value = batch_dim.dim_value
                            batch_dim.dim_value = fixed_batch_size
                            needs_fix = True
                    elif batch_dim.HasField('dim_param'):
                        # Fix if it's a parameter (should be fixed value for fixed batch models)
                        old_value = batch_dim.dim_param
                        batch_dim.ClearField('dim_param')
                        batch_dim.dim_value = fixed_batch_size
                        needs_fix = True
                    
                    if needs_fix:
                        tensors_fixed += 1
                        if tensors_fixed <= 20:  # Log first 20 for debugging
                            print(f"   Fixed tensor '{vi.name}': batch {old_value} -> {fixed_batch_size}")
            
            print(f"✅ Fixed {tensors_fixed} intermediate tensor batch dimensions")
            
            # Re-run shape inference after manual fixes to propagate changes
            if tensors_fixed > 0:
                print(f"🔧 Re-running shape inference to propagate fixed batch dimensions...")
                try:
                    inferred_model = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
                    model = inferred_model
                    print("✅ Second shape inference completed")
                    
                    # NEW: Identify scale groups and assign unique spatial parameters
                    # This prevents ONNX Runtime from trying to reuse buffers across different scales
                    # which have different spatial dimensions (40x40 vs 80x80, etc.)
                    print(f"\n🔧 STEP 1: Identifying scale groups from network structure...")
                    tensor_to_scale_local = {}
                    scale_groups_local = {}
                    try:
                        # Use graph-based method to identify scales from network structure
                        tensor_to_scale_local, scale_groups_local, tensor_scale_depth_local = identify_scale_groups(model)
                        
                        if not tensor_to_scale_local:
                            print("   ⚠️  Graph-based method failed, trying shape-inference-based method...")
                            tensor_to_scale_local, scale_groups_local = identify_scales_from_shape_inference(model)
                            tensor_scale_depth_local = None
                        
                        if tensor_to_scale_local:
                            print(f"\n🔧 STEP 2: Assigning unique spatial dimension parameters to scale groups...")
                            tensors_updated = assign_unique_spatial_parameters(model, tensor_to_scale_local, fixed_batch_size, tensor_scale_depth_local)
                            
                            if tensors_updated > 0:
                                print(f"✅ Successfully assigned unique spatial parameters to {len(scale_groups_local)} scale groups")
                                
                                # Re-run shape inference one more time to propagate unique parameters
                                print(f"🔧 Re-running shape inference to propagate unique spatial parameters...")
                                try:
                                    inferred_model = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
                                    model = inferred_model
                                    print("✅ Third shape inference completed")
                                except Exception as e:
                                    print(f"⚠️  Third shape inference failed: {e}")
                                    print("⚠️  Continuing anyway - unique parameters should still work")
                            else:
                                print(f"⚠️  No tensors updated - scale groups may not have been identified correctly")
                        else:
                            print(f"⚠️  No scale groups identified - using default '?' parameters")
                    except Exception as e:
                        print(f"⚠️  Scale group identification failed: {e}")
                        print("⚠️  Continuing with default '?' parameters (may cause buffer reuse issues)")
                        import traceback
                        traceback.print_exc()
                    
                    # CRITICAL FIX: Replace 'unk__' dimensions with appropriate parameters
                    # If scale groups were identified, use scale-specific parameters
                    # Otherwise, use '?' (dynamic) as fallback
                    print(f"\n🔧 Fixing 'unk__' dimensions...")
                    unk_fixed = 0
                    
                    for vi in model.graph.value_info:
                        if vi.type.tensor_type.shape:
                            dims = vi.type.tensor_type.shape.dim
                            for dim_index, dim in enumerate(dims):
                                # Check if dimension is an unknown dimension created by shape inference
                                if dim.HasField('dim_param'):
                                    dim_param = dim.dim_param
                                    if dim_param.startswith('unk__'):
                                        # Get scale ID for this tensor
                                        scale_id = tensor_to_scale_local.get(vi.name) if tensor_to_scale_local else None
                                        
                                        # Determine which parameter to use
                                        if scale_id is not None and dim_index >= 2 and len(dims) >= 4:
                                            # Spatial dimension (H or W) - use scale-specific parameter
                                            if dim_index == 2:  # H dimension
                                                new_param = f'H{scale_id}'
                                            elif dim_index == 3:  # W dimension
                                                new_param = f'W{scale_id}'
                                            else:
                                                new_param = '?'  # Fallback
                                        else:
                                            # Batch/channel dimension or unknown scale - use '?'
                                            new_param = '?'
                                        
                                        dim.ClearField('dim_param')
                                        dim.dim_param = new_param
                                        unk_fixed += 1
                                        if unk_fixed <= 10:
                                            print(f"   Fixed '{vi.name}' dim[{dim_index}]: '{dim_param}' -> '{new_param}'")
                    
                    if unk_fixed > 0:
                        print(f"✅ Fixed {unk_fixed} 'unk__' dimensions")
                    else:
                        print(f"✅ No 'unk__' dimensions found (good!)")
                    
                except Exception as e:
                    print(f"⚠️  Second shape inference failed: {e}")
                    print("⚠️  Continuing anyway - manual fixes should be sufficient")
            
    except Exception as e:
        print(f"⚠️  Shape inference failed: {e}")
        print("⚠️  Continuing without shape inference (may cause DirectML issues)")
        # Continue anyway - some models might work without shape inference
    
    # Validate modified model BEFORE saving
    print("🔍 Validating modified model...")
    try:
        onnx.checker.check_model(model)
        print("✅ Modified model validation passed")
    except Exception as e:
        print(f"❌ Modified model validation failed: {e}")
        print("❌ Not saving invalid model. Original model is unchanged.")
        return False
    
    # Save modified model
    print(f"💾 Saving modified model: {output_path}")
    try:
        onnx.save(model, output_path)
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False
    
    # Verify saved model
    print("🔍 Verifying saved model...")
    if not validate_model(output_path):
        print(f"❌ Saved model failed validation! File may be corrupted.")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"🗑️  Removed corrupted file: {output_path}")
        return False
    
    # Final verification of batch dimension
    verify_model = onnx.load(output_path)
    new_batch = verify_model.graph.input[0].type.tensor_type.shape.dim[0]
    if fixed_batch_size is not None:
        if new_batch.HasField('dim_value') and new_batch.dim_value == fixed_batch_size:
            print(f"✅ Final verification: Batch dimension is now fixed={fixed_batch_size}")
        else:
            print(f"❌ Final verification failed: Expected batch={fixed_batch_size}, got {new_batch.dim_value if new_batch.HasField('dim_value') else 'dynamic'}")
            return False
    else:
        if new_batch.HasField('dim_param'):
            print(f"✅ Final verification: Batch dimension is now '{new_batch.dim_param}'")
        else:
            print(f"❌ Final verification failed: Batch dimension is still fixed")
            return False
    
    print(f"\n🎉 Success!")
    print(f"   Original: {model_path} (unchanged)")
    print(f"   Modified: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Modify InsightFace ONNX model for batch processing')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Fixed batch size (default: 512). Use 0 for dynamic batch.')
    args = parser.parse_args()
    
    print("=" * 60)
    print("InsightFace Model Batch Modification Tool")
    print("=" * 60)
    print()
    
    # Determine batch size (0 = dynamic, >0 = fixed)
    fixed_batch_size = args.batch_size if args.batch_size > 0 else None
    
    if fixed_batch_size:
        print(f"📊 Mode: Fixed batch size = {fixed_batch_size}")
    else:
        print("📊 Mode: Dynamic batch dimension")
    print("📝 Output: Always named with '_modified' suffix (e.g., det_10g_modified.onnx)")
    print("   This allows changing batch sizes without hardcoding model names.")
    print()
    
    # Find the model (always uses "_modified" suffix)
    model_path, output_path = find_model_path(fixed_batch_size)
    
    if not model_path:
        print("❌ Model not found!")
        print("\n📥 Please download InsightFace models first:")
        print("   python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')\"")
        print("\n💡 Expected locations:")
        print(f"   - ~/.insightface/models/buffalo_l/det_10g.onnx")
        print(f"   - ~/.insightface/models/buffalo_l/detection.onnx")
        sys.exit(1)
    
    print(f"📁 Original model: {model_path}")
    print(f"📁 Modified model: {output_path}")
    print("=" * 60)
    print()
    
    success = modify_model_for_batching(model_path, output_path, backup=True, fixed_batch_size=fixed_batch_size)
    
    if success:
        print("\n✅ Modification complete!")
        print("💡 The original model is unchanged and safe.")
        print("💡 The modified model is saved as: det_10g_modified.onnx")
        print("💡 You can re-run this script with different batch sizes anytime.")
        print("\n🔄 Next steps:")
        if fixed_batch_size:
            print(f"   1. Set gpu_target_batch={fixed_batch_size} in config.py")
            print(f"   2. Restart the GPU worker to load the modified model")
            print(f"   3. Ensure batches are exactly {fixed_batch_size} images (or padded)")
            print("   4. Verify batch processing works with DirectML")
        else:
            print("   1. Restart the GPU worker to load the modified model")
            print("   2. Check logs for '[SCRFD] Dynamic batch dimension confirmed'")
            print("   3. Verify batch processing works with batches > 1")
        sys.exit(0)
    else:
        print("\n❌ Modification failed!")
        print("💡 The original model is unchanged and safe.")
        print("💡 You can re-run this script to try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()


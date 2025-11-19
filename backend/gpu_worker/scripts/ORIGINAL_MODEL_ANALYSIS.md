# Original Model Dynamic Dimension Analysis

## Summary

Analysis of how the original SCRFD model (`det_10g.onnx`) handles dynamic dimensions across tensors to understand why it works with batch=1 but our modified model fails with batch=32.

## Key Findings

### 1. Version Synchronization
- ✅ **Fixed**: ONNX Runtime versions are now synchronized
  - `onnxruntime`: 1.23.0
  - `onnxruntime-directml`: 1.23.0
- ❌ **Result**: Version sync did NOT fix the buffer reuse issue

### 2. Original Model Dynamic Dimension Usage

#### Input Shape
- **Original**: `[1, 3, '?', '?']`
  - Batch: Fixed at 1
  - Spatial: Dynamic (`'?'`)

#### Key Tensor Shapes

**Resize_124:**
- Input `382`: `[1, 56, '?', '?']` - Batch fixed at 1
- Sizes `399`: `[4]` - Constant shape specification
- Output `401`: `['?', '?', '?', '?']` - **All dimensions dynamic**

**Resize_144:**
- Input `402`: `['?', 56, '?', '?']` - **Batch is dynamic `'?'`** ⚠️
- Sizes `419`: `[4]` - Constant shape specification  
- Output `421`: `['?', '?', '?', '?']` - **All dimensions dynamic**

### 3. Critical Difference

**Original Model:**
- Input has batch=1 (fixed)
- But intermediate tensors can have batch=`'?'` (dynamic)
- Resize outputs have **all dimensions dynamic** (`['?', '?', '?', '?']`)
- This allows ONNX Runtime flexibility in buffer allocation

**Modified Model:**
- Input has batch=32 (fixed)
- All intermediate tensors have batch=32 (fixed)
- Resize outputs have batch=32 fixed, spatial dynamic (`[32, '?', '?', '?']`)
- This causes ONNX Runtime to be more aggressive with buffer reuse

### 4. Why Original Model Works (batch=1)

1. **Small batch size**: ONNX Runtime is less aggressive with buffer reuse for batch=1
2. **Mixed batch dimensions**: Some tensors have batch=1, others have batch=`'?'`
3. **Fully dynamic outputs**: Resize outputs are `['?', '?', '?', '?']`, giving maximum flexibility
4. **Less buffer reuse**: Single-image inference doesn't expose multi-scale shape conflicts

### 5. Why Modified Model Fails (batch=32)

1. **Large batch size**: ONNX Runtime is more aggressive with buffer reuse for larger batches
2. **Fixed batch dimensions**: All tensors have batch=32, making buffer allocation more rigid
3. **Partial dynamic outputs**: Resize outputs are `[32, '?', '?', '?']`, constraining buffer reuse
4. **Buffer reuse conflicts**: ONNX Runtime tries to reuse buffers for tensors with:
   - Same batch dimension (32) ✅
   - Same channel dimension (56) ✅
   - **Different spatial dimensions** (40x40 vs 80x80) ❌

### 6. Root Cause

The error message reveals the issue:
```
Shape mismatch attempting to re-use buffer. {32,56,40,40} != {32,56,80,80}
```

**Problem**: ONNX Runtime sees two tensors with:
- Same batch: 32
- Same channels: 56
- Different spatial: 40x40 vs 80x80

Since both use the same dynamic parameter `'?'` for spatial dimensions, ONNX Runtime expects them to have the same size when reusing buffers. But they're from different scales in the multi-scale architecture, so they have different sizes at runtime.

### 7. Multi-Scale Architecture

The SCRFD model uses **multi-scale feature extraction**:
- **Scale 1**: 40x40 spatial dimensions
- **Scale 2**: 80x80 spatial dimensions  
- **Scale 3**: 160x160 spatial dimensions
- All scales use the same dynamic parameter `'?'` for spatial dimensions

**Original model (batch=1)**: ONNX Runtime doesn't aggressively reuse buffers, so the conflict doesn't appear.

**Modified model (batch=32)**: ONNX Runtime aggressively reuses buffers, exposing the multi-scale shape inconsistency.

## Insights

### Why Original Model Uses Mixed Batch Dimensions

The original model has:
- Input: `[1, 3, '?', '?']` - batch fixed at 1
- Intermediate tensor `382`: `[1, 56, '?', '?']` - batch fixed at 1
- Intermediate tensor `402`: `['?', 56, '?', '?']` - **batch dynamic** ⚠️
- Resize outputs: `['?', '?', '?', '?']` - **all dimensions dynamic**

This suggests that shape inference in the original model allows some flexibility in batch dimension propagation, even though the input is fixed at batch=1.

### Why This Matters

When we fix batch=32 throughout the model, we lose this flexibility. ONNX Runtime becomes more aggressive about buffer reuse because:
1. All batch dimensions are fixed (32), so it can pre-allocate buffers
2. All tensors with the same batch/channel dimensions look "reusable"
3. But different scales have different spatial dimensions, causing conflicts

## Potential Solutions

### Solution A: Keep Some Batch Dimensions Dynamic (Complex)

Allow some intermediate tensors to have dynamic batch dimensions, similar to the original model. This would require:
1. Identifying which tensors can safely have dynamic batch
2. Modifying only those tensors that need fixed batch
3. Ensuring shape inference propagates correctly

**Pros**: Might preserve ONNX Runtime flexibility
**Cons**: Complex to implement, might not work with batch=32

### Solution B: Use Different Dimension Parameters for Different Scales (Most Promising)

Assign unique dimension parameters to different scales:
- Scale 1: `'H1'`, `'W1'` for spatial dimensions
- Scale 2: `'H2'`, `'W2'` for spatial dimensions
- Scale 3: `'H3'`, `'W3'` for spatial dimensions

This would tell ONNX Runtime that these are different dimensions and shouldn't be reused.

**Pros**: Properly models multi-scale architecture
**Cons**: Complex to implement, requires identifying all scale groups

### Solution C: Disable Buffer Reuse (Already Tested - Didn't Work)

We tested disabling `enable_mem_pattern` and `enable_cpu_mem_arena`, but it didn't fix the issue.

**Status**: ❌ Tested - Did not work

### Solution D: Use Original Model with Single-Image Processing

Revert to batch=1 model and process images one at a time.

**Pros**: Works reliably
**Cons**: Loses batch processing performance benefits

## Next Steps

1. ✅ **Version synchronization**: Complete
2. ✅ **Original model analysis**: Complete
3. **Implement Solution B**: Use different dimension parameters for different scales
4. **Test Solution A**: Try keeping some batch dimensions dynamic
5. **Fallback to Solution D**: Use single-image processing if needed

## Files Created

- `inspect_original_model_shapes.py`: Analyzes original model's dynamic dimension usage
- `compare_model_shapes.py`: Compares original vs modified model shapes
- `ORIGINAL_MODEL_ANALYSIS.md`: This document




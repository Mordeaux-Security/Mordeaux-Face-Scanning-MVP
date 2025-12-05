# DirectML Resize_144 Error Troubleshooting

## Problem Summary

**Error**: `RuntimeException: Non-zero status code returned while running Resize node. Name:'Resize_144'`

**Error Code**: `8007023E {Application Error}`

**Root Cause**: Model has inconsistent spatial dimensions in multi-scale feature extraction paths. ONNX Runtime attempts to reuse buffers but fails because tensors have different spatial sizes (40x40 vs 80x80) while using the same dynamic dimension parameter (`'?'`).

## Key Findings

### 1. Version Information

- `onnxruntime`: 1.23.0 ✅ (fixed)
- `onnxruntime-directml`: 1.23.0 ✅ (fixed)
- **Status**: Versions synchronized, but did NOT fix the issue

### 2. Error Occurs on Both CPU and DirectML

- **CPU inference**: FAILS with shape mismatch error
- **DirectML inference**: FAILS with DirectML error
- **Conclusion**: This is NOT a DirectML-specific issue - it's a model shape inference problem

### 3. Error Message Analysis

```
Shape mismatch attempting to re-use buffer. {32,56,40,40} != {32,56,80,80}
Validate usage of dim_value (values should be > 0) and dim_param
(all values with the same string should equate to the same size) in shapes in the model.
```

### 4. Resize_144 Node Configuration

- **Mode**: `nearest`
- **Coordinate transformation mode**: `asymmetric`
- **Nearest mode**: `floor`
- **Inputs**: `['402', '392', '392', '419']`
  - Input 0 (data): `402` - shape `[32, 56, '?', '?']` ✅
  - Input 1: `392` (initializer/constant)
  - Input 2: `392` (duplicate - likely scales)
  - Input 3: `419` (Concat output - likely sizes tensor)

## Root Cause Analysis

The SCRFD model uses **multi-scale feature extraction**:

- Different scales have different spatial dimensions (40x40, 80x80, 160x160, etc.)
- All scales use the same dynamic spatial dimension parameter (`'?'`)
- ONNX Runtime expects all tensors using the same `dim_param` to have the same size
- When ONNX Runtime tries to reuse buffers for efficiency, it fails because different scales have different sizes

## Why Original Model Works (batch=1)

The original model (batch=1, dynamic spatial) works because:

1. ONNX Runtime is less aggressive with buffer reuse for small batches
2. Buffer pre-allocation is less aggressive
3. Single-image inference doesn't expose the multi-scale shape inconsistency

## Why Modified Model Fails (batch=32)

The modified model (batch=32, dynamic spatial) fails because:

1. ONNX Runtime is more aggressive with buffer reuse for larger batches
2. Buffer pre-allocation tries to reuse buffers across different scales
3. The shape inconsistency becomes apparent when buffers are reused

## Potential Solutions

### Solution 1: Disable Buffer Pre-allocation (TESTED - DID NOT WORK)

Disable memory pattern optimization to prevent buffer reuse:

```python
sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = False  # Disable buffer reuse
sess_options.enable_cpu_mem_arena = False  # Disable CPU memory arena
```

**Status**: ❌ Tested - Still fails with same error
**Conclusion**: Issue is not just buffer reuse optimization, but fundamental shape inconsistency in model definition

### Solution 2: Use Different Dimension Parameters for Different Scales

Instead of using `'?'` for all spatial dimensions, use unique parameters:

- Scale 1: `'H1'`, `'W1'`
- Scale 2: `'H2'`, `'W2'`
- Scale 3: `'H3'`, `'W3'`

**Pros**: Properly models multi-scale architecture
**Cons**: Complex to implement, requires understanding model architecture

### Solution 3: Fix Spatial Dimensions to Concrete Values

Since images are always 640x640 after letterbox, we could trace through the network and fix all spatial dimensions to concrete values.

**Pros**: Eliminates shape inconsistency
**Cons**: Defeats purpose of dynamic spatial dimensions, complex to implement correctly

### Solution 4: Use Original Model with Single-Image Processing

Revert to batch=1 model and process images one at a time.

**Pros**: Works reliably
**Cons**: Loses batch processing performance benefits

### Solution 5: Try Different ONNX Runtime Version

Some users report that specific versions (e.g., 1.16.1) handle dynamic shapes better.

**Pros**: Simple to test
**Cons**: May not fix the issue, could introduce other problems

## Test Results

### ✅ Original Model (batch=1) Works

- Original `det_10g.onnx` with batch=1 works correctly
- Confirms model architecture is correct
- Issue is specifically with batched processing

### 📊 Original Model Analysis

**Key Finding**: Original model uses **mixed batch dimensions**:

- Input: `[1, 3, '?', '?']` - batch fixed at 1
- Some tensors: `[1, 56, '?', '?']` - batch fixed at 1
- Other tensors: `['?', 56, '?', '?']` - **batch dynamic** ⚠️
- Resize outputs: `['?', '?', '?', '?']` - **all dimensions dynamic**

**Why This Matters**:

- Original model has flexibility in batch dimension propagation
- Resize outputs are fully dynamic, giving ONNX Runtime maximum flexibility
- Modified model fixes batch=32 throughout, making buffer allocation rigid
- ONNX Runtime becomes aggressive with buffer reuse for fixed batch dimensions

**See**: `ORIGINAL_MODEL_ANALYSIS.md` for detailed analysis

### ❌ Solution 1 Tested: Buffer Pre-allocation Disabled

- Disabled `enable_mem_pattern` and `enable_cpu_mem_arena`
- Still fails with same error
- Confirms issue is not buffer reuse optimization, but fundamental shape inconsistency

## Recommended Next Steps

1. ✅ **Solution 1 tested** - Did not fix the issue
2. ✅ **Original model verified** - Works with batch=1
3. **Priority: Implement Solution 2** (different dim_params for different scales) - Most promising
4. **Fallback: Solution 4** (single-image processing) - Reliable but slower
5. **Alternative: Solution 5** (try older ONNX Runtime version) - Quick test

## Current Status

- ✅ Model modification approach is correct (batch fixed, spatial dynamic)
- ✅ Input shape is correct: `[32, 3, '?', '?']`
- ❌ Model has multi-scale shape inconsistency that prevents batched inference
- ❌ Issue affects both CPU and DirectML providers

## Files Modified

- `backend/gpu_worker/scripts/modify_onnx_batch.py`: Fixed to keep spatial dimensions dynamic
- `backend/gpu_worker/scripts/inspect_onnx_model.py`: Enhanced Resize node inspection
- `backend/gpu_worker/scripts/diagnose_directml.py`: Created diagnostic tool

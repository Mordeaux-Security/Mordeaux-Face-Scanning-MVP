# Batched Face Detection Implementation Notes

## Implementation Path: Path A

**Choice:** Path A - Add batched detector path inside GPU worker service

**Rationale:**
- Detector already lives in GPU worker (`backend/gpu_worker/worker.py`)
- InsightFace uses SCRFD internally (accessible via `_face_app.models['detection']`)
- Minimal disruption: keep existing single-image fallback, add batched path behind feature flag
- Centralized control: all batching logic in one place

## Architecture

### Components

1. **Detector Package** (`backend/gpu_worker/detectors/`)
   - `common.py`: Letterbox preprocessing, NMS, coordinate mapping
   - `scrfd_onnx.py`: SCRFD batched detection implementation
   - `__init__.py`: Package exports

2. **GPU Worker** (`backend/gpu_worker/worker.py`)
   - Batched detection path (when `BATCHED_DETECTOR_ENABLED=true`)
   - Single-image fallback (original logic, always available)
   - Metrics tracking and structured logging
   - Auto-backoff on DML/driver errors

3. **GPU Scheduler** (`backend/new_crawler/gpu_scheduler.py`)
   - Already implemented: centralized batching, 2-inflight limit, min launch gap
   - Handles pacing at the crawler level (not detector level)

### Flow

```
Extractor → gpu:inbox (per-image items) → GPU Scheduler → GPU Worker
                                                              ↓
                                                      Batched Detector
                                                      (if enabled)
                                                              ↓
                                                      Results keyed by phash
```

## Configuration

### Environment Variables

```bash
# Enable batched detector
BATCHED_DETECTOR_ENABLED=true

# Detector batch configuration
DETECT_TARGET_BATCH=16          # Target batch size for detector
DETECT_MAX_WAIT_MS=12           # Max wait before early launch
DETECT_MIN_LAUNCH_MS=180        # Minimum ms between launches (Windows/AMD stability)

# Detector model parameters
DETECT_INPUT_SIZE=640           # Letterbox size
DETECT_SCORE_THR=0.5            # Detection score threshold
DETECT_NMS_IOU=0.4              # NMS IoU threshold

# GPU Scheduler (crawler level)
GPU_TARGET_BATCH=32             # Target batch size for GPU scheduler
GPU_MAX_WAIT_MS=12              # Max wait before early launch
GPU_MIN_LAUNCH_MS=200           # Minimum ms between batch launches
GPU_INBOX_KEY=gpu:inbox         # Redis queue key
```

### Safe Defaults for Windows + AMD (DirectML)

- `DETECT_TARGET_BATCH=16`: Balanced batch size for AMD GPUs
- `DETECT_MIN_LAUNCH_MS=180`: Prevents "double launch <100ms" crashes on Windows
- `GPU_MIN_LAUNCH_MS=200`: Additional safety margin at scheduler level

## Dynamic Batch Dimension

### Current State

InsightFace models loaded via `FaceAnalysis.prepare()` should support dynamic batch dimensions by default. The detector verifies this on load and logs a warning if batch=1 is fixed.

### Exporting Dynamic-N ONNX

If your ONNX model has a fixed batch dimension (e.g., batch=1), you need to export it with dynamic batch:

```python
import onnx
from onnx.tools import update_model_dims

# Load model
model = onnx.load("model.onnx")

# Make batch dimension dynamic
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch'
model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'batch'

# Save
onnx.save(model, "model_dynamic.onnx")
```

### Verification

Check model input shape:
```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_shape = session.get_inputs()[0].shape
print(input_shape)  # Should show [-1, 3, 640, 640] or ['batch', 3, 640, 640]
```

## Logging and Metrics

### Structured Logs

**On LAUNCH:**
```
[BATCHED-DET-LAUNCH] stage=det, batch_id=1234567890, size=16, inflight=1, interlaunch_ms=185.2
```

**On COMPLETE:**
```
[BATCHED-DET-COMPLETE] batch_id=1234567890, size=16, latency_ms=245.3, t_det_forward=180.1ms, t_det_post=15.2ms, inflight=0, faces=42
```

**Periodic Summary (every 10s):**
```
[BATCHED-DET-SUMMARY] batches=50, images=800, avg_batch=16.0, p50_latency=240.5ms, p95_latency=320.1ms, p50_det_forward=175.2ms, p95_det_forward=250.3ms, p50_interlaunch=185.0ms, p95_interlaunch=220.5ms, p50_idle=5.0ms, p95_idle=15.2ms, inflight_max=2
```

### Pace Violations

If `interlaunch_ms < MIN_LAUNCH_MS`, you'll see:
```
[BATCHED-DET] Pace violation: interlaunch_ms=150.0ms < MIN=180ms
```

This indicates the scheduler is launching batches too quickly - increase `DETECT_MIN_LAUNCH_MS` or `GPU_MIN_LAUNCH_MS`.

### Verification Checklist

To confirm batching is working:

1. **GPU logs show paced launches:**
   - No `interlaunch_ms < MIN_LAUNCH_MS` warnings
   - `inflight ≤ 2` (enforced by GPU scheduler)

2. **Detector forward runs with N>1:**
   - Look for `[BATCHED-DET-LAUNCH] size=N` where N > 1 (most of the time when queue is non-empty)
   - `t_det_forward` time should scale better than linearly (batched is faster per image)

3. **Performance improvement:**
   - Compare `t_det_forward` per image: batched should be lower
   - Overall images/sec should not regress; ideally improves on batches ≥1000 images

4. **No correctness regressions:**
   - Number of faces per image statistically matches baseline within tolerance

## Auto-Backoff

The system automatically reduces batch size and increases launch gap on DML/driver errors:

- **Trigger:** Errors containing "dml", "directml", "driver", or "gpu"
- **Action:**
  - Reduce `DETECT_TARGET_BATCH` by 25% (minimum 4)
  - Increase `DETECT_MIN_LAUNCH_MS` by +50ms
- **Duration:** 30 seconds
- **Restoration:** Automatically restores original values after backoff period

**Logs:**
```
[AUTO-BACKOFF] Activated: target_batch=12 (was 16), min_launch_ms=230 (was 180), duration=30.0s
[AUTO-BACKOFF] Restored: target_batch=16, min_launch_ms=180
```

## Troubleshooting

### Batched Detector Not Loading

**Symptoms:** Logs show "Failed to load batched detector"

**Checks:**
1. Verify `BATCHED_DETECTOR_ENABLED=true`
2. Check that InsightFace model loaded successfully
3. Verify detection model exists: `_face_app.models['detection']`
4. Check ONNX session has providers configured

### Batched Detection Always Falls Back

**Symptoms:** Always seeing "Batched detector failed, falling back to single-image"

**Possible Causes:**
1. Batch size ≤ 1 (batched path only activates for batch_size > 1)
2. ONNX model has fixed batch=1 (check logs for warning)
3. SCRFD decode failing (check error messages)

**Solutions:**
- Ensure batch size > 1
- Export model with dynamic batch dimension
- Check SCRFD decode implementation matches your model's output format

### Poor GPU Utilization

**Symptoms:** GPU utilization is low, batches processing slowly

**Checks:**
1. Verify DirectML is actually being used (check logs for "GPU acceleration enabled")
2. Check batch sizes: should see `size=N` where N ≥ TARGET_BATCH most of the time
3. Check interlaunch times: should be ≥ MIN_LAUNCH_MS
4. Reduce `MIN_LAUNCH_MS` if too conservative (but keep ≥180ms on Windows/AMD)

### Pace Violations

**Symptoms:** Frequent "Pace violation" warnings

**Solutions:**
- Increase `DETECT_MIN_LAUNCH_MS` or `GPU_MIN_LAUNCH_MS`
- Check if GPU scheduler is properly enforcing limits
- Reduce `TARGET_BATCH` to process faster and reduce pressure

### Auto-Backoff Triggering Frequently

**Symptoms:** Frequent "[AUTO-BACKOFF] Activated" messages

**Possible Causes:**
1. AMD driver issues (update drivers)
2. DirectML compatibility issues
3. Memory pressure causing GPU errors
4. Batch size too large for GPU

**Solutions:**
- Update AMD GPU drivers
- Reduce `DETECT_TARGET_BATCH` permanently (set as default)
- Increase `DETECT_MIN_LAUNCH_MS` permanently
- Check GPU memory usage

## Testing

### Unit Tests

See `backend/gpu_worker/tests/test_scrfd_detector.py` for:
- Letterbox preprocessing
- NMS functionality
- Coordinate mapping
- Batch shape handling

### Bench Script

Run `backend/scripts/bench_detector.py` to compare:
- Single-image mode vs. batched mode
- Performance metrics (imgs/s, t_det_forward, t_det_post)
- Correctness validation

## Next Steps

1. **Refine SCRFD Decode:** Current implementation uses simplified decode. May need to align with exact InsightFace SCRFD output format if results don't match single-image path.

2. **Optimize Batch Sizes:** Tune `DETECT_TARGET_BATCH` and `GPU_TARGET_BATCH` based on actual GPU performance characteristics.

3. **Add Embedder Batching:** Once detector batching is stable, consider batching the embedder (ArcFace) as well for further speedup.


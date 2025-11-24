"""
DirectML Diagnostic Script

Checks ONNX Runtime/DirectML installation, versions, and compatibility.
"""

import sys
import subprocess
import os
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_package_version(package_name):
    """Check if a package is installed and return its version."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
        return None
    except Exception as e:
        return f"Error: {e}"

def check_onnxruntime_providers():
    """Check available ONNX Runtime providers."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        version = ort.__version__
        return providers, version
    except ImportError:
        return None, None
    except Exception as e:
        return f"Error: {e}", None

def check_gpu_info():
    """Check GPU information using Windows WMI."""
    try:
        import wmi
        c = wmi.WMI()
        gpus = []
        for gpu in c.Win32_VideoController():
            gpus.append({
                'name': gpu.Name or 'Unknown',
                'driver_version': gpu.DriverVersion or 'Unknown',
                'driver_date': str(gpu.DriverDate) if gpu.DriverDate else 'Unknown'
            })
        return gpus
    except ImportError:
        # Try alternative method using dxdiag or PowerShell
        try:
            result = subprocess.run(
                ['powershell', '-Command', 'Get-WmiObject Win32_VideoController | Select-Object Name, DriverVersion | Format-Table -AutoSize'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout
            return "Could not retrieve GPU info (pywin32 not installed)"
        except:
            return "Could not retrieve GPU info (pywin32 not installed)"
    except Exception as e:
        return f"Error: {e}"

def test_cpu_inference(model_path):
    """Test model inference with CPU provider."""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print(f"\n[TEST] Testing model with CPU provider: {model_path}")
        
        if not os.path.exists(model_path):
            return f"Model not found: {model_path}"
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # Test: Disable buffer pre-allocation to prevent buffer reuse issues with multi-scale models
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        print(f"   ✅ Session created with providers: {session.get_providers()}")
        
        # Get input info
        input_info = session.get_inputs()[0]
        print(f"   Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        
        # Create test input (batch=32, 640x640)
        test_shape = (32, 3, 640, 640)
        test_input = np.random.randn(*test_shape).astype(np.float32)
        print(f"   Test input shape: {test_input.shape}, dtype: {test_input.dtype}")
        
        # Try inference
        print(f"   Attempting inference...")
        try:
            outputs = session.run(None, {input_info.name: test_input})
            print(f"   ✅ CPU inference succeeded!")
            print(f"   Outputs: {len(outputs)} outputs")
            for i, out in enumerate(outputs):
                print(f"      Output {i}: shape={out.shape}, dtype={out.dtype}")
            return "SUCCESS"
        except Exception as e:
            print(f"   ❌ CPU inference failed: {type(e).__name__}: {e}")
            return f"FAILED: {e}"
            
    except Exception as e:
        return f"Error: {e}"

def test_dml_inference(model_path):
    """Test model inference with DirectML provider."""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print(f"\n[TEST] Testing model with DirectML provider: {model_path}")
        
        if not os.path.exists(model_path):
            return f"Model not found: {model_path}"
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # Test: Disable buffer pre-allocation to prevent buffer reuse issues with multi-scale models
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        actual_providers = session.get_providers()
        print(f"   ✅ Session created with providers: {actual_providers}")
        
        if actual_providers[0] != 'DmlExecutionProvider':
            return f"WARNING: DirectML not used, got {actual_providers[0]}"
        
        # Get input info
        input_info = session.get_inputs()[0]
        print(f"   Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        
        # Create test input (batch=32, 640x640)
        test_shape = (32, 3, 640, 640)
        test_input = np.random.randn(*test_shape).astype(np.float32)
        print(f"   Test input shape: {test_input.shape}, dtype: {test_input.dtype}")
        
        # Try inference
        print(f"   Attempting inference...")
        try:
            outputs = session.run(None, {input_info.name: test_input})
            print(f"   ✅ DirectML inference succeeded!")
            print(f"   Outputs: {len(outputs)} outputs")
            for i, out in enumerate(outputs):
                print(f"      Output {i}: shape={out.shape}, dtype={out.dtype}")
            return "SUCCESS"
        except Exception as e:
            print(f"   ❌ DirectML inference failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return f"FAILED: {e}"
            
    except Exception as e:
        return f"Error: {e}"

def main():
    """Run all diagnostic checks."""
    print("=" * 80)
    print("DirectML Diagnostic Tool")
    print("=" * 80)
    print()
    
    # Check Python version
    print(f"[SYSTEM] Python version: {sys.version}")
    print(f"[SYSTEM] Python executable: {sys.executable}")
    print()
    
    # Check ONNX Runtime packages
    print("[PACKAGES] Checking ONNX Runtime packages...")
    onnxruntime_version = check_package_version('onnxruntime')
    onnxruntime_dml_version = check_package_version('onnxruntime-directml')
    
    if onnxruntime_version:
        print(f"   onnxruntime: {onnxruntime_version}")
    else:
        print(f"   onnxruntime: Not installed")
    
    if onnxruntime_dml_version:
        print(f"   onnxruntime-directml: {onnxruntime_dml_version}")
    else:
        print(f"   onnxruntime-directml: Not installed (may be included in onnxruntime)")
    
    print()
    
    # Check available providers
    print("[PROVIDERS] Checking ONNX Runtime providers...")
    providers, ort_version = check_onnxruntime_providers()
    if providers:
        print(f"   ONNX Runtime version: {ort_version}")
        print(f"   Available providers: {providers}")
        if 'DmlExecutionProvider' in providers:
            print(f"   ✅ DirectML provider available")
        else:
            print(f"   ❌ DirectML provider NOT available")
    else:
        print(f"   ❌ ONNX Runtime not installed or error: {providers}")
    
    print()
    
    # Check GPU info
    print("[GPU] Checking GPU information...")
    gpu_info = check_gpu_info()
    if isinstance(gpu_info, list):
        for i, gpu in enumerate(gpu_info):
            print(f"   GPU {i+1}: {gpu['name']}")
            print(f"      Driver version: {gpu['driver_version']}")
            print(f"      Driver date: {gpu['driver_date']}")
    else:
        print(f"   {gpu_info}")
    
    print()
    
    # Test model inference
    home = os.path.expanduser("~/.insightface")
    model_path = os.path.join(home, "models", "buffalo_l", "det_10g_modified.onnx")
    
    if os.path.exists(model_path):
        print("[MODEL] Testing model inference...")
        
        # Test CPU first
        cpu_result = test_cpu_inference(model_path)
        
        # Test DirectML
        dml_result = test_dml_inference(model_path)
        
        print()
        print("[RESULTS]")
        print(f"   CPU inference: {cpu_result}")
        print(f"   DirectML inference: {dml_result}")
    else:
        print(f"[MODEL] Model not found: {model_path}")
        print("   Skipping inference tests")
    
    print()
    print("=" * 80)
    print("Diagnostic complete")
    print("=" * 80)

if __name__ == "__main__":
    main()


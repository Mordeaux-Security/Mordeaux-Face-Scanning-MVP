# GPU Detection and Environment Setup Script for Windows

Write-Host "Detecting GPU hardware..."

# Run GPU detection
python backend/scripts/detect_gpu.py > $env:TEMP/gpu_detection.log 2>&1

# Check detection results
$gpuLog = Get-Content "$env:TEMP/gpu_detection.log" -Raw

if ($gpuLog -match "GPU Type: amd") {
    Write-Host "AMD GPU detected - configuring for ROCm"
    $env:GPU_TYPE = "amd"
    $env:GPU_DRIVER = "rocm"
    $env:GPU_BACKEND = "rocm"
} elseif ($gpuLog -match "GPU Type: nvidia") {
    Write-Host "NVIDIA GPU detected - configuring for CUDA"
    $env:GPU_TYPE = "nvidia"
    $env:GPU_DRIVER = "nvidia"
    $env:GPU_BACKEND = "cuda"
} elseif ($gpuLog -match "GPU Type: intel") {
    Write-Host "Intel GPU detected - configuring for Intel GPU"
    $env:GPU_TYPE = "intel"
    $env:GPU_DRIVER = "intel"
    $env:GPU_BACKEND = "cpu"
} else {
    Write-Host "No compatible GPU detected - using CPU-only mode"
    $env:GPU_TYPE = "cpu"
    $env:GPU_DRIVER = "cpu"
    $env:GPU_BACKEND = "cpu"
}

Write-Host "GPU Configuration:"
Write-Host "  Type: $env:GPU_TYPE"
Write-Host "  Driver: $env:GPU_DRIVER"
Write-Host "  Backend: $env:GPU_BACKEND"

# Write to .env file
Add-Content -Path ".env" -Value "GPU_TYPE=$env:GPU_TYPE"
Add-Content -Path ".env" -Value "GPU_DRIVER=$env:GPU_DRIVER"
Add-Content -Path ".env" -Value "GPU_BACKEND=$env:GPU_BACKEND"

Write-Host "GPU configuration written to .env file"


# Quick Smoke Test for Mordeaux Face Scanning MVP
# Enhanced PowerShell script to test basic connectivity and routing with latency metrics

param(
    [string]$Host = "localhost",
    [int]$Port = 80,
    [int]$Samples = 10
)

Write-Host "🧪 Quick Smoke Test - Mordeaux Face Scanning MVP" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

$TotalTests = 0
$PassedTests = 0
$FailedTests = 0
$LatencyData = @{}

# Helper function to calculate percentiles
function Calculate-Percentile {
    param(
        [array]$Data,
        [int]$Percentile
    )
    if ($Data.Count -eq 0) { return 0 }
    $sorted = $Data | Sort-Object
    $index = [Math]::Floor(($Percentile / 100) * ($sorted.Count - 1))
    return $sorted[$index]
}

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Url,
        [int]$ExpectedStatus = 200,
        [bool]$TrackLatency = $false
    )
    
    $script:TotalTests++
    Write-Host "Testing: $Name" -ForegroundColor Blue
    
    $latencies = @()
    $successCount = 0
    
    # Run multiple samples for latency tracking
    $samplesToRun = if ($TrackLatency) { $Samples } else { 1 }
    
    for ($i = 1; $i -le $samplesToRun; $i++) {
        try {
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 10
            $stopwatch.Stop()
            $latency = $stopwatch.ElapsedMilliseconds
            
            if ($response.StatusCode -eq $ExpectedStatus) {
                $successCount++
                $latencies += $latency
            }
        } catch {
            # Count as failure
        }
    }
    
    $successRate = if ($samplesToRun -gt 0) { $successCount / $samplesToRun } else { 0 }
    
    if ($successRate -eq 1.0) {
        $statusText = "✅ PASS: $Name - Status: $ExpectedStatus"
        if ($TrackLatency -and $latencies.Count -gt 0) {
            $p50 = Calculate-Percentile $latencies 50
            $p95 = Calculate-Percentile $latencies 95
            $avg = ($latencies | Measure-Object -Average).Average
            $statusText += " - P50: ${p50}ms, P95: ${p95}ms, Avg: ${avg}ms"
            $LatencyData[$Name] = @{
                P50 = $p50
                P95 = $p95
                Avg = $avg
                Samples = $latencies.Count
            }
        }
        Write-Host $statusText -ForegroundColor Green
        $script:PassedTests++
    } else {
        Write-Host "❌ FAIL: $Name - Success rate: $($successRate * 100)%" -ForegroundColor Red
        $script:FailedTests++
    }
}

# Test basic connectivity with latency tracking
Write-Host "🔍 Testing basic connectivity..." -ForegroundColor Yellow
Test-Endpoint "Nginx Main" "http://${Host}:${Port}/" -TrackLatency $true
Test-Endpoint "Backend Health" "http://${Host}:${Port}/api/health" -TrackLatency $true
Test-Endpoint "Pipeline Health" "http://${Host}:${Port}/face-pipeline/health" -TrackLatency $true

# Test CORS headers
Write-Host "`n🌐 Testing CORS headers..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://${Host}:${Port}/api/health" -UseBasicParsing
    if ($response.Headers['Access-Control-Allow-Origin']) {
        Write-Host "✅ PASS: CORS headers present" -ForegroundColor Green
        $script:PassedTests++
    } else {
        Write-Host "❌ FAIL: CORS headers missing" -ForegroundColor Red
        $script:FailedTests++
    }
    $script:TotalTests++
} catch {
    Write-Host "❌ FAIL: CORS test failed - $($_.Exception.Message)" -ForegroundColor Red
    $script:FailedTests++
    $script:TotalTests++
}

# Test port mapping
Write-Host "`n🔌 Testing port mapping..." -ForegroundColor Yellow
Test-Endpoint "Direct Backend" "http://localhost:8000/health"
Test-Endpoint "Direct Frontend" "http://localhost:3000/"
Test-Endpoint "Direct Pipeline" "http://localhost:8001/health"

# Summary
Write-Host "`n📊 Test Summary" -ForegroundColor Cyan
Write-Host "===============" -ForegroundColor Cyan
Write-Host "Total tests: $script:TotalTests"
Write-Host "Passed: $script:PassedTests" -ForegroundColor Green
Write-Host "Failed: $script:FailedTests" -ForegroundColor Red

# Latency Summary
if ($LatencyData.Count -gt 0) {
    Write-Host "`n⚡ Latency Metrics" -ForegroundColor Cyan
    Write-Host "==================" -ForegroundColor Cyan
    foreach ($endpoint in $LatencyData.Keys) {
        $data = $LatencyData[$endpoint]
        $samplesText = "$($data.Samples) samples"
        Write-Host "$endpoint : P50=$($data.P50)ms, P95=$($data.P95)ms, Avg=$($data.Avg)ms ($samplesText)" -ForegroundColor White
    }
}

if ($script:FailedTests -eq 0) {
    Write-Host "`n🎉 All tests passed! The proxy is working correctly." -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n❌ Some tests failed. Please check the configuration." -ForegroundColor Red
    exit 1
}

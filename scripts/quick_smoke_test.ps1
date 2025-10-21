# Quick Smoke Test for Mordeaux Face Scanning MVP
# Simple PowerShell script to test basic connectivity and routing

param(
    [string]$Host = "localhost",
    [int]$Port = 80
)

Write-Host "🧪 Quick Smoke Test - Mordeaux Face Scanning MVP" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

$TotalTests = 0
$PassedTests = 0
$FailedTests = 0

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Url,
        [int]$ExpectedStatus = 200
    )
    
    $script:TotalTests++
    Write-Host "Testing: $Name" -ForegroundColor Blue
    
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq $ExpectedStatus) {
            Write-Host "✅ PASS: $Name - Status: $($response.StatusCode)" -ForegroundColor Green
            $script:PassedTests++
        } else {
            Write-Host "❌ FAIL: $Name - Expected: $ExpectedStatus, Got: $($response.StatusCode)" -ForegroundColor Red
            $script:FailedTests++
        }
    } catch {
        Write-Host "❌ FAIL: $Name - Error: $($_.Exception.Message)" -ForegroundColor Red
        $script:FailedTests++
    }
}

# Test basic connectivity
Write-Host "🔍 Testing basic connectivity..." -ForegroundColor Yellow
Test-Endpoint "Nginx Main" "http://${Host}:${Port}/"
Test-Endpoint "Backend Health" "http://${Host}:${Port}/api/health"
Test-Endpoint "Pipeline Health" "http://${Host}:${Port}/face-pipeline/health"

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

# Test performance
Write-Host "`n⚡ Testing performance..." -ForegroundColor Yellow
try {
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-WebRequest -Uri "http://${Host}:${Port}/api/health" -UseBasicParsing
    $stopwatch.Stop()
    $latency = $stopwatch.ElapsedMilliseconds
    
    if ($latency -lt 200) {
        Write-Host "✅ PASS: Health endpoint latency ${latency}ms (under 200ms)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  WARN: Health endpoint latency ${latency}ms (exceeds 200ms)" -ForegroundColor Yellow
    }
    $script:TotalTests++
    $script:PassedTests++
} catch {
    Write-Host "❌ FAIL: Performance test failed - $($_.Exception.Message)" -ForegroundColor Red
    $script:FailedTests++
    $script:TotalTests++
}

# Summary
Write-Host "`n📊 Test Summary" -ForegroundColor Cyan
Write-Host "===============" -ForegroundColor Cyan
Write-Host "Total tests: $script:TotalTests"
Write-Host "Passed: $script:PassedTests" -ForegroundColor Green
Write-Host "Failed: $script:FailedTests" -ForegroundColor Red

if ($script:FailedTests -eq 0) {
    Write-Host "`n🎉 All tests passed! The proxy is working correctly." -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n❌ Some tests failed. Please check the configuration." -ForegroundColor Red
    exit 1
}

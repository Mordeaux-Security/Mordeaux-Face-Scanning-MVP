# Comprehensive Integration Test for Mordeaux Face Scanning MVP
# Full end-to-end testing with detailed metrics collection

param(
    [string]$Host = "localhost",
    [int]$Port = 80,
    [int]$Samples = 10,
    [string]$OutputDir = "qa"
)

$StartTime = Get-Date
$Timestamp = $StartTime.ToString("yyyyMMdd_HHmmss")

Write-Host "🧪 Comprehensive Integration Test - Mordeaux Face Scanning MVP" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Started: $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
Write-Host ""

# Test results tracking
$script:TotalTests = 0
$script:PassedTests = 0
$script:FailedTests = 0
$script:LatencyData = @{}
$script:TestResults = @()
$script:ContainerStatus = @{}

# Helper functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[PASS] $Message" -ForegroundColor Green
    $script:PassedTests++
}

function Write-Error {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
    $script:FailedTests++
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

# Calculate percentiles
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

# Test endpoint with comprehensive metrics
function Test-Endpoint {
    param(
        [string]$TestName,
        [string]$Url,
        [int]$ExpectedStatus = 200,
        [bool]$TrackLatency = $false,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = $null
    )
    
    $script:TotalTests++
    Write-Info "Running: $TestName"
    
    $latencies = @()
    $successCount = 0
    $errorMessages = @()
    
    # Run multiple samples for latency tracking
    $samplesToRun = if ($TrackLatency) { $Samples } else { 1 }
    
    for ($i = 1; $i -le $samplesToRun; $i++) {
        try {
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            
            $requestParams = @{
                Uri = $Url
                UseBasicParsing = $true
                TimeoutSec = 30
                Method = $Method
            }
            
            if ($Headers.Count -gt 0) {
                $requestParams.Headers = $Headers
            }
            
            if ($Body) {
                $requestParams.Body = $Body
                $requestParams.ContentType = "application/json"
            }
            
            $response = Invoke-WebRequest @requestParams
            $stopwatch.Stop()
            $latency = $stopwatch.ElapsedMilliseconds
            
            if ($response.StatusCode -eq $ExpectedStatus) {
                $successCount++
                $latencies += $latency
            } else {
                $errorMessages += "Expected $ExpectedStatus, got $($response.StatusCode)"
            }
        } catch {
            $errorMessages += $_.Exception.Message
        }
    }
    
    $successRate = if ($samplesToRun -gt 0) { $successCount / $samplesToRun } else { 0 }
    
    # Store test result
    $testResult = @{
        Name = $TestName
        Url = $Url
        ExpectedStatus = $ExpectedStatus
        SuccessRate = $successRate
        SuccessCount = $successCount
        TotalSamples = $samplesToRun
        ErrorMessages = $errorMessages
        Timestamp = Get-Date
    }
    
    if ($TrackLatency -and $latencies.Count -gt 0) {
        $p50 = Calculate-Percentile $latencies 50
        $p95 = Calculate-Percentile $latencies 95
        $p99 = Calculate-Percentile $latencies 99
        $avg = ($latencies | Measure-Object -Average).Average
        $min = ($latencies | Measure-Object -Minimum).Minimum
        $max = ($latencies | Measure-Object -Maximum).Maximum
        
        $testResult.Latency = @{
            P50 = $p50
            P95 = $p95
            P99 = $p99
            Avg = $avg
            Min = $min
            Max = $max
            Samples = $latencies.Count
        }
        
        $script:LatencyData[$TestName] = $testResult.Latency
    }
    
    $script:TestResults += $testResult
    
    if ($successRate -eq 1.0) {
        $statusText = "✅ PASS: $TestName - Status: $ExpectedStatus"
        if ($TrackLatency -and $latencies.Count -gt 0) {
            $statusText += " - P50: $($testResult.Latency.P50)ms, P95: $($testResult.Latency.P95)ms, P99: $($testResult.Latency.P99)ms"
        }
        Write-Success $statusText
    } else {
        Write-Error "$TestName - Success rate: $($successRate * 100)%"
        if ($errorMessages.Count -gt 0) {
            Write-Warning "Errors: $($errorMessages -join '; ')"
        }
    }
}

# Check Docker container status
function Test-ContainerStatus {
    Write-Info "Checking Docker container status..."
    
    try {
        $containers = docker ps --filter "name=mordeaux" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Out-String
        Write-Host $containers
        
        # Parse container status
        $containerLines = $containers -split "`n" | Where-Object { $_ -match "mordeaux" }
        foreach ($line in $containerLines) {
            if ($line -match "mordeaux-face-scanning-mvp-(\w+)-1\s+(.+)") {
                $serviceName = $matches[1]
                $status = $matches[2]
                $script:ContainerStatus[$serviceName] = $status
            }
        }
        
        Write-Success "Container status check completed"
    } catch {
        Write-Error "Failed to check container status: $($_.Exception.Message)"
    }
}

# Test health endpoints
function Test-HealthEndpoints {
    Write-Info "Testing health endpoints..."
    
    # Test nginx proxy
    Test-Endpoint "Nginx Proxy" "http://${Host}:${Port}/" -TrackLatency $true
    
    # Test backend health (if available)
    Test-Endpoint "Backend Health" "http://${Host}:${Port}/api/health" -TrackLatency $true
    Test-Endpoint "Backend Ready" "http://${Host}:${Port}/api/ready" -TrackLatency $true
    
    # Test face-pipeline health (if available)
    Test-Endpoint "Face Pipeline Health" "http://${Host}:${Port}/face-pipeline/health" -TrackLatency $true
    Test-Endpoint "Face Pipeline Ready" "http://${Host}:${Port}/face-pipeline/ready" -TrackLatency $true
    
    # Test direct service access
    Test-Endpoint "Direct Backend" "http://localhost:8000/health" -TrackLatency $true
    Test-Endpoint "Direct Frontend" "http://localhost:3000/" -TrackLatency $true
    Test-Endpoint "Direct Pipeline" "http://localhost:8001/health" -TrackLatency $true
}

# Test search endpoints
function Test-SearchEndpoints {
    Write-Info "Testing search endpoints..."
    
    # Test search by image (if backend is available)
    $testImagePath = "face-pipeline/samples/test_image.jpg"
    if (Test-Path $testImagePath) {
        try {
            $boundary = [System.Guid]::NewGuid().ToString()
            $LF = "`r`n"
            $bodyLines = (
                "--$boundary",
                "Content-Disposition: form-data; name=`"file`"; filename=`"test_image.jpg`"",
                "Content-Type: image/jpeg$LF",
                [System.IO.File]::ReadAllBytes($testImagePath),
                "--$boundary--$LF"
            ) -join $LF
            
            Test-Endpoint "Search by Image" "http://${Host}:${Port}/api/search_face?top_k=5&threshold=0.25" -Method "POST" -Headers @{"X-Tenant-ID"="test-tenant"; "Content-Type"="multipart/form-data; boundary=$boundary"} -Body $bodyLines -TrackLatency $true
        } catch {
            Write-Warning "Search by image test skipped - test image not available or service not ready"
        }
    } else {
        Write-Warning "Search by image test skipped - test image not found at $testImagePath"
    }
    
    # Test search by vector (if face-pipeline is available)
    $vectorPayload = @{
        vector = @(0.1, 0.2, 0.3, 0.4, 0.5) * 100  # Mock 500-dimensional vector
        top_k = 5
        threshold = 0.25
    } | ConvertTo-Json
    
    Test-Endpoint "Search by Vector" "http://${Host}:${Port}/face-pipeline/api/v1/search" -Method "POST" -Headers @{"X-Tenant-ID"="test-tenant"; "Content-Type"="application/json"} -Body $vectorPayload -TrackLatency $true
}

# Test CORS and headers
function Test-CorsHeaders {
    Write-Info "Testing CORS headers..."
    
    try {
        $response = Invoke-WebRequest -Uri "http://${Host}:${Port}/api/health" -UseBasicParsing
        $corsHeaders = @(
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Methods", 
            "Access-Control-Allow-Headers"
        )
        
        $missingHeaders = @()
        foreach ($header in $corsHeaders) {
            if (-not $response.Headers[$header]) {
                $missingHeaders += $header
            }
        }
        
        if ($missingHeaders.Count -eq 0) {
            Write-Success "All CORS headers present"
        } else {
            Write-Error "Missing CORS headers: $($missingHeaders -join ', ')"
        }
    } catch {
        Write-Error "CORS test failed: $($_.Exception.Message)"
    }
}

# Test concurrent requests
function Test-ConcurrentRequests {
    Write-Info "Testing concurrent request handling..."
    
    $concurrentRequests = 5
    $jobs = @()
    
    for ($i = 1; $i -le $concurrentRequests; $i++) {
        $jobs += Start-Job -ScriptBlock {
            param($Url)
            try {
                $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 10
                return @{
                    Success = $true
                    StatusCode = $response.StatusCode
                    Latency = $response.Headers["X-Response-Time"]
                }
            } catch {
                return @{
                    Success = $false
                    Error = $_.Exception.Message
                }
            }
        } -ArgumentList "http://${Host}:${Port}/"
    }
    
    $results = $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job
    
    $successCount = ($results | Where-Object { $_.Success }).Count
    $successRate = $successCount / $concurrentRequests
    
    if ($successRate -ge 0.8) {
        Write-Success "Concurrent requests: $successCount/$concurrentRequests successful ($([Math]::Round($successRate * 100))%)"
    } else {
        Write-Error "Concurrent requests: $successCount/$concurrentRequests successful ($([Math]::Round($successRate * 100))%)"
    }
}

# Generate comprehensive report
function Generate-Report {
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    $reportContent = @"
# Integration Test Report - Mordeaux Face Scanning MVP

**Test Execution Summary**
- **Date**: $($StartTime.ToString('yyyy-MM-dd'))
- **Time**: $($StartTime.ToString('HH:mm:ss'))
- **Duration**: $($Duration.TotalMinutes.ToString('F2')) minutes
- **Total Tests**: $script:TotalTests
- **Passed**: $script:PassedTests
- **Failed**: $script:FailedTests
- **Success Rate**: $([Math]::Round(($script:PassedTests / $script:TotalTests) * 100, 2))%

## Container Status
"@

    foreach ($container in $script:ContainerStatus.Keys) {
        $status = $script:ContainerStatus[$container]
        $reportContent += "`n- **$container**: $status"
    }

    $reportContent += @"

## Test Results
"@

    foreach ($result in $script:TestResults) {
        $reportContent += "`n### $($result.Name)"
        $reportContent += "`n- **URL**: $($result.Url)"
        $reportContent += "`n- **Success Rate**: $([Math]::Round($result.SuccessRate * 100, 2))%"
        $reportContent += "`n- **Expected Status**: $($result.ExpectedStatus)"
        
        if ($result.Latency) {
            $reportContent += "`n- **Latency Metrics**:"
            $reportContent += "`n  - P50: $($result.Latency.P50)ms"
            $reportContent += "`n  - P95: $($result.Latency.P95)ms"
            $reportContent += "`n  - P99: $($result.Latency.P99)ms"
            $reportContent += "`n  - Average: $($result.Latency.Avg)ms"
            $reportContent += "`n  - Min: $($result.Latency.Min)ms"
            $reportContent += "`n  - Max: $($result.Latency.Max)ms"
            $reportContent += "`n  - Samples: $($result.Latency.Samples)"
        }
        
        if ($result.ErrorMessages.Count -gt 0) {
            $reportContent += "`n- **Errors**: $($result.ErrorMessages -join '; ')"
        }
    }

    $reportContent += @"

## Latency Summary
"@

    if ($script:LatencyData.Count -gt 0) {
        foreach ($endpoint in $script:LatencyData.Keys) {
            $data = $script:LatencyData[$endpoint]
            $reportContent += "`n- **$endpoint**: P50=${data.P50}ms, P95=${data.P95}ms, P99=${data.P99}ms, Avg=${data.Avg}ms"
        }
    }

    $reportContent += @"

## Recommendations
"@

    if ($script:FailedTests -eq 0) {
        $reportContent += "`n✅ All tests passed! The system is working correctly."
    } else {
        $reportContent += "`n❌ Some tests failed. Please check the following:"
        $reportContent += "`n- Verify all Docker containers are running"
        $reportContent += "`n- Check service logs for errors"
        $reportContent += "`n- Ensure all dependencies are healthy"
    }

    # Save report
    $reportPath = Join-Path $OutputDir "smoke-test-results-${Timestamp}.md"
    $reportContent | Out-File -FilePath $reportPath -Encoding UTF8
    
    Write-Info "Report saved to: $reportPath"
    return $reportPath
}

# Main test execution
function Main {
    # Create output directory if it doesn't exist
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }
    
    # Run all tests
    Test-ContainerStatus
    Test-HealthEndpoints
    Test-SearchEndpoints
    Test-CorsHeaders
    Test-ConcurrentRequests
    
    # Generate report
    $reportPath = Generate-Report
    
    # Print summary
    Write-Host ""
    Write-Host "📊 Test Summary" -ForegroundColor Cyan
    Write-Host "===============" -ForegroundColor Cyan
    Write-Host "Total tests: $script:TotalTests"
    Write-Host "Passed: $script:PassedTests" -ForegroundColor Green
    Write-Host "Failed: $script:FailedTests" -ForegroundColor Red
    Write-Host "Success rate: $([Math]::Round(($script:PassedTests / $script:TotalTests) * 100, 2))%"
    Write-Host "Report: $reportPath"
    
    if ($script:FailedTests -eq 0) {
        Write-Host ""
        Write-Success "All integration tests passed! 🎉"
        exit 0
    } else {
        Write-Host ""
        Write-Error "Some integration tests failed. Please check the report for details."
        exit 1
    }
}

# Run main function
Main

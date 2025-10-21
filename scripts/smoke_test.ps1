# Mordeaux Face Scanning MVP - Proxy Smoke Tests (PowerShell)
# Tests Nginx reverse proxy routing and API endpoints

param(
    [string]$NginxHost = "localhost",
    [int]$NginxPort = 80,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [int]$PipelinePort = 8001
)

# Test results tracking
$script:TotalTests = 0
$script:PassedTests = 0
$script:FailedTests = 0

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

# Test function
function Test-Endpoint {
    param(
        [string]$TestName,
        [string]$Url,
        [int]$ExpectedStatus,
        [int]$MaxLatency = 0
    )
    
    $script:TotalTests++
    Write-Info "Running: $TestName"
    
    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 30
        $stopwatch.Stop()
        $latency = $stopwatch.ElapsedMilliseconds
        
        if ($response.StatusCode -eq $ExpectedStatus) {
            if ($MaxLatency -gt 0 -and $latency -gt $MaxLatency) {
                Write-Warning "$TestName - Status OK ($($response.StatusCode)) but latency too high (${latency}ms > ${MaxLatency}ms)"
            } else {
                Write-Success "$TestName - Status: $($response.StatusCode), Latency: ${latency}ms"
            }
        } else {
            Write-Error "$TestName - Expected status $ExpectedStatus, got $($response.StatusCode)"
        }
    } catch {
        Write-Error "$TestName - Request failed: $($_.Exception.Message)"
    }
}

# Check if services are running
function Test-Services {
    Write-Info "Checking if services are running..."
    
    # Check if Docker containers are running
    try {
        $dockerPs = docker-compose ps
        if ($dockerPs -notmatch "Up") {
            Write-Error "Docker services are not running. Please run 'make start' first."
            exit 1
        }
    } catch {
        Write-Error "Failed to check Docker services: $($_.Exception.Message)"
        exit 1
    }
    
    # Check if ports are accessible
    $services = @(
        @{Name="nginx"; Port=$NginxPort},
        @{Name="backend"; Port=$BackendPort},
        @{Name="frontend"; Port=$FrontendPort},
        @{Name="pipeline"; Port=$PipelinePort}
    )
    
    foreach ($service in $services) {
        try {
            $tcpClient = New-Object System.Net.Sockets.TcpClient
            $tcpClient.Connect("localhost", $service.Port)
            $tcpClient.Close()
            Write-Success "$($service.Name) service is accessible on port $($service.Port)"
        } catch {
            Write-Error "$($service.Name) service is not accessible on port $($service.Port)"
        }
    }
}

# Test Nginx routing
function Test-NginxRouting {
    Write-Info "Testing Nginx reverse proxy routing..."
    
    # Test frontend routing
    Test-Endpoint "Frontend routing" "http://${NginxHost}:${NginxPort}/" 200
    
    # Test backend API routing
    Test-Endpoint "Backend API routing" "http://${NginxHost}:${NginxPort}/api/health" 200
    
    # Test face-pipeline routing
    Test-Endpoint "Face Pipeline routing" "http://${NginxHost}:${NginxPort}/face-pipeline/health" 200
}

# Test CORS headers
function Test-CorsHeaders {
    Write-Info "Testing CORS headers..."
    
    try {
        # Test CORS preflight request
        $headers = @{
            'Origin' = 'http://localhost:3000'
            'Access-Control-Request-Method' = 'GET'
        }
        $response = Invoke-WebRequest -Uri "http://${NginxHost}:${NginxPort}/api/health" -Method OPTIONS -Headers $headers -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Success "CORS preflight request successful"
        } else {
            Write-Error "CORS preflight request failed with status $($response.StatusCode)"
        }
        
        # Test CORS headers in response
        $response = Invoke-WebRequest -Uri "http://${NginxHost}:${NginxPort}/api/health" -UseBasicParsing
        if ($response.Headers['Access-Control-Allow-Origin']) {
            Write-Success "CORS headers present in API response"
        } else {
            Write-Error "CORS headers missing in API response"
        }
    } catch {
        Write-Error "CORS test failed: $($_.Exception.Message)"
    }
}

# Test port mapping
function Test-PortMapping {
    Write-Info "Testing port mapping..."
    
    # Test direct backend access
    Test-Endpoint "Direct backend access" "http://localhost:${BackendPort}/health" 200
    
    # Test direct frontend access
    Test-Endpoint "Direct frontend access" "http://localhost:${FrontendPort}/" 200
    
    # Test direct pipeline access
    Test-Endpoint "Direct pipeline access" "http://localhost:${PipelinePort}/health" 200
}

# Test API endpoints through proxy
function Test-ApiEndpoints {
    Write-Info "Testing API endpoints through proxy..."
    
    # Test health endpoints
    Test-Endpoint "Backend health through proxy" "http://${NginxHost}:${NginxPort}/api/health" 200 200
    
    Test-Endpoint "Pipeline health through proxy" "http://${NginxHost}:${NginxPort}/face-pipeline/health" 200 200
    
    # Test ready endpoints
    Test-Endpoint "Backend ready through proxy" "http://${NginxHost}:${NginxPort}/api/ready" 200 200
    
    Test-Endpoint "Pipeline ready through proxy" "http://${NginxHost}:${NginxPort}/face-pipeline/ready" 503 200
    
    # Test search endpoint (should return 405 Method Not Allowed for GET)
    Test-Endpoint "Search endpoint through proxy" "http://${NginxHost}:${NginxPort}/api/v1/search" 405
}

# Test performance
function Test-Performance {
    Write-Info "Testing performance requirements..."
    
    try {
        # Test health endpoint latency
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-WebRequest -Uri "http://${NginxHost}:${NginxPort}/api/health" -UseBasicParsing
        $stopwatch.Stop()
        $latency = $stopwatch.ElapsedMilliseconds
        
        if ($latency -lt 200) {
            Write-Success "Health endpoint latency: ${latency}ms (under 200ms requirement)"
        } else {
            Write-Warning "Health endpoint latency: ${latency}ms (exceeds 200ms requirement)"
        }
        
        # Test multiple concurrent requests
        Write-Info "Testing concurrent request handling..."
        $concurrentRequests = 5
        $successCount = 0
        
        $jobs = @()
        for ($i = 1; $i -le $concurrentRequests; $i++) {
            $jobs += Start-Job -ScriptBlock {
                param($Url)
                try {
                    $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 10
                    return $response.StatusCode -eq 200
                } catch {
                    return $false
                }
            } -ArgumentList "http://${NginxHost}:${NginxPort}/api/health"
        }
        
        $results = $jobs | Wait-Job | Receive-Job
        $jobs | Remove-Job
        
        $successCount = ($results | Where-Object { $_ -eq $true }).Count
        
        if ($successCount -eq $concurrentRequests) {
            Write-Success "Concurrent request handling: $successCount/$concurrentRequests successful"
        } else {
            Write-Warning "Concurrent request handling: $successCount/$concurrentRequests successful"
        }
    } catch {
        Write-Error "Performance test failed: $($_.Exception.Message)"
    }
}

# Test error handling
function Test-ErrorHandling {
    Write-Info "Testing error handling..."
    
    # Test 404 handling
    Test-Endpoint "404 error handling" "http://${NginxHost}:${NginxPort}/api/nonexistent" 404
    
    # Test invalid method
    try {
        $response = Invoke-WebRequest -Uri "http://${NginxHost}:${NginxPort}/api/health" -Method DELETE -UseBasicParsing
        if ($response.StatusCode -eq 405) {
            Write-Success "Invalid method handling - Status: $($response.StatusCode)"
        } else {
            Write-Error "Invalid method handling - Expected 405, got $($response.StatusCode)"
        }
    } catch {
        if ($_.Exception.Response.StatusCode -eq 405) {
            Write-Success "Invalid method handling - Status: 405"
        } else {
            Write-Error "Invalid method handling failed: $($_.Exception.Message)"
        }
    }
}

# Test Nginx configuration
function Test-NginxConfig {
    Write-Info "Testing Nginx configuration..."
    
    try {
        # Test if Nginx is serving the correct content
        $frontendResponse = Invoke-WebRequest -Uri "http://${NginxHost}:${NginxPort}/" -UseBasicParsing
        if ($frontendResponse.Content -match "html|<!DOCTYPE") {
            Write-Success "Nginx is serving frontend content"
        } else {
            Write-Error "Nginx is not serving frontend content properly"
        }
        
        # Test if API routes are properly proxied
        $apiResponse = Invoke-WebRequest -Uri "http://${NginxHost}:${NginxPort}/api/health" -UseBasicParsing
        if ($apiResponse.Content -match "status|healthy") {
            Write-Success "Nginx is properly proxying API requests"
        } else {
            Write-Error "Nginx is not properly proxying API requests"
        }
    } catch {
        Write-Error "Nginx configuration test failed: $($_.Exception.Message)"
    }
}

# Main test execution
function Main {
    Write-Host "🧪 Mordeaux Face Scanning MVP - Proxy Smoke Tests" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check prerequisites
    Test-Services
    
    Write-Host ""
    Write-Info "Starting smoke tests..."
    Write-Host ""
    
    # Run all tests
    Test-NginxRouting
    Test-CorsHeaders
    Test-PortMapping
    Test-ApiEndpoints
    Test-Performance
    Test-ErrorHandling
    Test-NginxConfig
    
    # Print summary
    Write-Host ""
    Write-Host "📊 Test Summary" -ForegroundColor Cyan
    Write-Host "===============" -ForegroundColor Cyan
    Write-Host "Total tests: $script:TotalTests"
    Write-Host "Passed: $script:PassedTests" -ForegroundColor Green
    Write-Host "Failed: $script:FailedTests" -ForegroundColor Red
    
    if ($script:FailedTests -eq 0) {
        Write-Host ""
        Write-Success "All smoke tests passed! 🎉"
        exit 0
    } else {
        Write-Host ""
        Write-Error "Some smoke tests failed. Please check the configuration."
        exit 1
    }
}

# Run main function
Main

"""
Test Suite for Mock Server - Phase 3
====================================

Comprehensive tests to validate mock server functionality.

Usage:
    python test_mock_server.py
"""

import requests
import json
import time
from typing import Dict, List

BASE_URL = "http://localhost:8000"

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_test(name: str):
    print(f"\n{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}TEST: {name}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'='*80}{Colors.ENDC}")


def print_pass(message: str):
    print(f"{Colors.GREEN}✓ PASS:{Colors.ENDC} {message}")


def print_fail(message: str):
    print(f"{Colors.RED}✗ FAIL:{Colors.ENDC} {message}")


def print_info(message: str):
    print(f"{Colors.YELLOW}ℹ INFO:{Colors.ENDC} {message}")


def test_health_check():
    """Test health check endpoint"""
    print_test("Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        
        if response.status_code == 200:
            print_pass("Health check endpoint responded")
            data = response.json()
            print_info(f"Service: {data.get('service')}")
            print_info(f"Version: {data.get('version')}")
            print_info(f"Status: {data.get('status')}")
            print_info(f"Available fixtures: {data.get('available_fixtures')}")
            return True
        else:
            print_fail(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_fail(f"Health check error: {e}")
        return False


def test_list_fixtures():
    """Test fixture listing endpoint"""
    print_test("List Fixtures")
    
    try:
        response = requests.get(f"{BASE_URL}/mock/fixtures")
        
        if response.status_code == 200:
            print_pass("Fixture listing endpoint responded")
            data = response.json()
            
            print_info("Available fixtures:")
            for name, info in data["fixtures"].items():
                print(f"  - {name}: {info['count']} results, scores {info['score_range']}")
            
            return True
        else:
            print_fail(f"Fixture listing failed with status {response.status_code}")
            return False
    except Exception as e:
        print_fail(f"Fixture listing error: {e}")
        return False


def test_search_basic():
    """Test basic search endpoint"""
    print_test("Basic Search (Medium Fixture)")
    
    try:
        # Create minimal multipart request
        files = {
            'image': ('test.jpg', b'fake-image-data', 'image/jpeg'),
            'tenant_id': (None, 'demo-tenant'),
            'top_k': (None, '10'),
            'threshold': (None, '0.75')
        }
        
        headers = {'X-Tenant-ID': 'demo-tenant'}
        
        response = requests.post(
            f"{BASE_URL}/api/v1/search",
            files=files,
            headers=headers
        )
        
        if response.status_code == 200:
            print_pass("Search endpoint responded")
            data = response.json()
            
            # Validate response structure
            assert "query" in data, "Missing 'query' field"
            assert "hits" in data, "Missing 'hits' field"
            assert "count" in data, "Missing 'count' field"
            
            print_pass("Response has correct structure")
            print_info(f"Query: tenant_id={data['query']['tenant_id']}, top_k={data['query']['top_k']}, threshold={data['query']['threshold']}")
            print_info(f"Results: {data['count']} hits")
            
            # Validate hits structure
            if data['hits']:
                hit = data['hits'][0]
                assert "face_id" in hit, "Hit missing 'face_id'"
                assert "score" in hit, "Hit missing 'score'"
                assert "payload" in hit, "Hit missing 'payload'"
                assert "thumb_url" in hit, "Hit missing 'thumb_url'"
                
                print_pass("Hit structure is correct")
                print_info(f"Top result: face_id={hit['face_id'][:20]}..., score={hit['score']}")
                print_info(f"Payload: site={hit['payload']['site']}, quality={hit['payload']['quality']}")
            
            return True
        else:
            print_fail(f"Search failed with status {response.status_code}")
            print_fail(f"Response: {response.text}")
            return False
    except Exception as e:
        print_fail(f"Search error: {e}")
        return False


def test_fixture_sizes():
    """Test different fixture sizes"""
    print_test("Fixture Sizes (Tiny, Medium, Large)")
    
    fixtures_to_test = ["tiny", "medium", "large"]
    expected_counts = {"tiny": 10, "medium": 200, "large": 2000}
    
    results = []
    
    for fixture in fixtures_to_test:
        try:
            files = {
                'image': ('test.jpg', b'fake-image-data', 'image/jpeg'),
                'tenant_id': (None, 'demo-tenant'),
                'top_k': (None, '10000'),  # Request all
                'threshold': (None, '0.0')  # No filtering
            }
            
            headers = {'X-Tenant-ID': 'demo-tenant'}
            
            response = requests.post(
                f"{BASE_URL}/api/v1/search?fixture={fixture}",
                files=files,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                count = data['count']
                
                if count == expected_counts[fixture]:
                    print_pass(f"{fixture}: {count} results (expected {expected_counts[fixture]})")
                    results.append(True)
                else:
                    print_fail(f"{fixture}: {count} results (expected {expected_counts[fixture]})")
                    results.append(False)
            else:
                print_fail(f"{fixture}: Request failed with status {response.status_code}")
                results.append(False)
        except Exception as e:
            print_fail(f"{fixture}: Error - {e}")
            results.append(False)
    
    return all(results)


def test_threshold_filtering():
    """Test threshold filtering"""
    print_test("Threshold Filtering")
    
    try:
        # Search with low threshold (should get many results)
        files_low = {
            'image': ('test.jpg', b'fake-image-data', 'image/jpeg'),
            'tenant_id': (None, 'demo-tenant'),
            'top_k': (None, '1000'),
            'threshold': (None, '0.5')
        }
        
        headers = {'X-Tenant-ID': 'demo-tenant'}
        
        response_low = requests.post(
            f"{BASE_URL}/api/v1/search?fixture=medium",
            files=files_low,
            headers=headers
        )
        
        # Search with high threshold (should get fewer results)
        files_high = {
            'image': ('test.jpg', b'fake-image-data', 'image/jpeg'),
            'tenant_id': (None, 'demo-tenant'),
            'top_k': (None, '1000'),
            'threshold': (None, '0.9')
        }
        
        response_high = requests.post(
            f"{BASE_URL}/api/v1/search?fixture=medium",
            files=files_high,
            headers=headers
        )
        
        if response_low.status_code == 200 and response_high.status_code == 200:
            data_low = response_low.json()
            data_high = response_high.json()
            
            count_low = data_low['count']
            count_high = data_high['count']
            
            print_info(f"Threshold 0.5: {count_low} results")
            print_info(f"Threshold 0.9: {count_high} results")
            
            if count_low > count_high:
                print_pass("Threshold filtering works correctly")
                
                # Verify all results are above threshold
                all_above = all(hit['score'] >= 0.9 for hit in data_high['hits'])
                if all_above:
                    print_pass("All high threshold results are >= 0.9")
                else:
                    print_fail("Some results are below threshold")
                    return False
                
                return True
            else:
                print_fail("Threshold filtering not working correctly")
                return False
        else:
            print_fail("Threshold filtering requests failed")
            return False
    except Exception as e:
        print_fail(f"Threshold filtering error: {e}")
        return False


def test_error_scenarios():
    """Test error scenarios"""
    print_test("Error Scenarios")
    
    scenarios = [
        ("no_results", 200, 0),  # Should return 200 with empty results
        ("api_error", 500, None),  # Should return 500
    ]
    
    results = []
    
    for scenario, expected_status, expected_count in scenarios:
        try:
            files = {
                'image': ('test.jpg', b'fake-image-data', 'image/jpeg'),
                'tenant_id': (None, 'demo-tenant')
            }
            
            headers = {'X-Tenant-ID': 'demo-tenant'}
            
            response = requests.post(
                f"{BASE_URL}/api/v1/search?error_scenario={scenario}",
                files=files,
                headers=headers
            )
            
            if response.status_code == expected_status:
                print_pass(f"{scenario}: Got expected status {expected_status}")
                
                if expected_count is not None:
                    data = response.json()
                    if data['count'] == expected_count:
                        print_pass(f"{scenario}: Got expected count {expected_count}")
                        results.append(True)
                    else:
                        print_fail(f"{scenario}: Expected count {expected_count}, got {data['count']}")
                        results.append(False)
                else:
                    results.append(True)
            else:
                print_fail(f"{scenario}: Expected status {expected_status}, got {response.status_code}")
                results.append(False)
        except Exception as e:
            print_fail(f"{scenario}: Error - {e}")
            results.append(False)
    
    return all(results)


def test_broken_urls():
    """Test fixtures with broken URLs"""
    print_test("Broken URLs (Errors Fixture)")
    
    try:
        files = {
            'image': ('test.jpg', b'fake-image-data', 'image/jpeg'),
            'tenant_id': (None, 'demo-tenant'),
            'top_k': (None, '100'),
            'threshold': (None, '0.0')
        }
        
        headers = {'X-Tenant-ID': 'demo-tenant'}
        
        response = requests.post(
            f"{BASE_URL}/api/v1/search?fixture=errors",
            files=files,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print_pass("Errors fixture loaded successfully")
            print_info(f"Results: {data['count']} hits")
            
            # Check that URLs contain error indicators
            broken_count = 0
            for hit in data['hits']:
                url = hit['thumb_url']
                if any(indicator in url for indicator in ['broken', 'expired', 'invalid', '404']):
                    broken_count += 1
            
            print_info(f"Broken URLs: {broken_count}/{data['count']}")
            
            if broken_count == data['count']:
                print_pass("All URLs in errors fixture are broken")
                return True
            else:
                print_fail(f"Expected all URLs to be broken, found {broken_count}/{data['count']}")
                return False
        else:
            print_fail(f"Errors fixture request failed with status {response.status_code}")
            return False
    except Exception as e:
        print_fail(f"Broken URLs test error: {e}")
        return False


def test_get_face_by_id():
    """Test get face by ID endpoint"""
    print_test("Get Face by ID")
    
    try:
        # First, get a face_id from search
        files = {
            'image': ('test.jpg', b'fake-image-data', 'image/jpeg'),
            'tenant_id': (None, 'demo-tenant')
        }
        
        headers = {'X-Tenant-ID': 'demo-tenant'}
        
        search_response = requests.post(
            f"{BASE_URL}/api/v1/search?fixture=tiny",
            files=files,
            headers=headers
        )
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            if search_data['hits']:
                face_id = search_data['hits'][0]['face_id']
                print_info(f"Testing with face_id: {face_id}")
                
                # Get face details
                face_response = requests.get(
                    f"{BASE_URL}/api/v1/faces/{face_id}?fixture=tiny",
                    headers={'X-Tenant-ID': 'demo-tenant'}
                )
                
                if face_response.status_code == 200:
                    print_pass("Get face by ID successful")
                    face_data = face_response.json()
                    
                    # Validate structure
                    assert "face_id" in face_data
                    assert "payload" in face_data
                    assert "thumb_url" in face_data
                    
                    print_pass("Face details have correct structure")
                    print_info(f"Face: {face_data['face_id']}")
                    print_info(f"Site: {face_data['payload']['site']}")
                    
                    return True
                else:
                    print_fail(f"Get face by ID failed with status {face_response.status_code}")
                    return False
            else:
                print_fail("Search returned no hits")
                return False
        else:
            print_fail("Search request failed")
            return False
    except Exception as e:
        print_fail(f"Get face by ID error: {e}")
        return False


def test_latency_simulation():
    """Test latency simulation"""
    print_test("Latency Simulation")
    
    try:
        # First disable latency
        config_response = requests.post(
            f"{BASE_URL}/mock/config",
            json={"simulate_latency": False}
        )
        
        if config_response.status_code != 200:
            print_fail("Failed to disable latency")
            return False
        
        # Test fast response
        start_time = time.time()
        files = {
            'image': ('test.jpg', b'fake-image-data', 'image/jpeg'),
            'tenant_id': (None, 'demo-tenant')
        }
        response = requests.post(
            f"{BASE_URL}/api/v1/search?fixture=tiny",
            files=files,
            headers={'X-Tenant-ID': 'demo-tenant'}
        )
        fast_time = time.time() - start_time
        
        # Enable latency
        config_response = requests.post(
            f"{BASE_URL}/mock/config",
            json={"simulate_latency": True, "min_latency_ms": 100, "max_latency_ms": 200}
        )
        
        # Test slow response
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/search?fixture=tiny",
            files=files,
            headers={'X-Tenant-ID': 'demo-tenant'}
        )
        slow_time = time.time() - start_time
        
        print_info(f"Fast response time: {fast_time*1000:.0f}ms")
        print_info(f"Slow response time: {slow_time*1000:.0f}ms")
        
        if slow_time > fast_time:
            print_pass("Latency simulation working")
            
            # Reset to default
            requests.post(
                f"{BASE_URL}/mock/config",
                json={"simulate_latency": True, "min_latency_ms": 50, "max_latency_ms": 300}
            )
            
            return True
        else:
            print_fail("Latency simulation not working")
            return False
    except Exception as e:
        print_fail(f"Latency simulation error: {e}")
        return False


def run_all_tests():
    """Run all tests and print summary"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}Mock Server Test Suite - Phase 3{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("List Fixtures", test_list_fixtures),
        ("Basic Search", test_search_basic),
        ("Fixture Sizes", test_fixture_sizes),
        ("Threshold Filtering", test_threshold_filtering),
        ("Error Scenarios", test_error_scenarios),
        ("Broken URLs", test_broken_urls),
        ("Get Face by ID", test_get_face_by_id),
        ("Latency Simulation", test_latency_simulation),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_fail(f"Test crashed: {e}")
            results.append((name, False))
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Print summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}Test Summary{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if result else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
        print(f"{status} - {name}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.ENDC}")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.ENDC}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Some tests failed{Colors.ENDC}\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    exit(exit_code)


#!/usr/bin/env python3
"""
Deployment Script for Verification-First Flow

Usage:
    python scripts/deploy.py dev      # Deploy to dev
    python scripts/deploy.py staging  # Deploy to staging
    python scripts/deploy.py prod     # Deploy to production

This script automates the deployment verification and smoke tests.
"""

import sys
import os
import time
import requests
import json
from typing import Dict, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def log_info(msg: str):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

def log_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")

def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def log_step(msg: str):
    print(f"\n{Colors.BLUE}=== {msg} ==={Colors.NC}")

# Environment configuration
ENV_CONFIGS = {
    'dev': {
        'qdrant_url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
        'api_url': os.getenv('API_URL', 'http://localhost:8001'),
        'timeout': 10,
    },
    'staging': {
        'qdrant_url': os.getenv('QDRANT_URL', 'http://qdrant-staging:6333'),
        'api_url': os.getenv('API_URL', 'http://api-staging.example.com'),
        'timeout': 15,
    },
    'prod': {
        'qdrant_url': os.getenv('QDRANT_URL', 'http://qdrant-prod:6333'),
        'api_url': os.getenv('API_URL', 'http://api-prod.example.com'),
        'timeout': 20,
    }
}

class DeploymentChecker:
    def __init__(self, env: str):
        if env not in ENV_CONFIGS:
            raise ValueError(f"Unknown environment: {env}. Use: dev, staging, prod")
        self.env = env
        self.config = ENV_CONFIGS[env]
        self.qdrant_url = self.config['qdrant_url']
        self.api_url = self.config['api_url']
        self.timeout = self.config['timeout']
        self.errors = []
        
    def check_qdrant(self) -> bool:
        """Check if Qdrant is accessible."""
        log_info("Checking Qdrant connectivity...")
        try:
            response = requests.get(f"{self.qdrant_url}/health", timeout=self.timeout)
            if response.status_code == 200:
                log_info(f"✓ Qdrant is accessible at {self.qdrant_url}")
                return True
            else:
                log_error(f"Qdrant returned status {response.status_code}")
                return False
        except Exception as e:
            log_error(f"Qdrant not accessible: {e}")
            return False
    
    def check_api(self) -> bool:
        """Check if API is accessible."""
        log_info("Checking API connectivity...")
        try:
            response = requests.get(f"{self.api_url}/api/v1/health", timeout=self.timeout)
            if response.status_code == 200:
                log_info(f"✓ API is accessible at {self.api_url}")
                return True
            else:
                log_warn(f"API returned status {response.status_code}")
                return False
        except Exception as e:
            log_warn(f"API not accessible: {e}")
            return False
    
    def check_collections(self) -> bool:
        """Check if required collections exist."""
        log_info("Checking Qdrant collections...")
        try:
            response = requests.get(f"{self.qdrant_url}/collections", timeout=self.timeout)
            if response.status_code != 200:
                log_warn(f"Could not fetch collections (status {response.status_code})")
                return False
            
            data = response.json()
            collections = [c['name'] for c in data.get('result', {}).get('collections', [])]
            
            # Check for faces_v1
            if 'faces_v1' in collections:
                log_info("✓ faces_v1 collection exists")
            else:
                log_warn("faces_v1 collection not found (will be created on startup)")
            
            # Check for identities_v1
            if 'identities_v1' in collections:
                log_info("✓ identities_v1 collection exists")
                return True
            else:
                log_warn("identities_v1 collection not found (will be created on startup)")
                return False
                
        except Exception as e:
            log_warn(f"Could not verify collections: {e}")
            return False
    
    def verify_collections_post_deploy(self) -> bool:
        """Verify collections after deployment."""
        log_info("Verifying collections after deployment...")
        time.sleep(5)  # Wait for collections to be created
        
        try:
            response = requests.get(f"{self.qdrant_url}/collections", timeout=self.timeout)
            if response.status_code != 200:
                log_error("Could not fetch collections")
                return False
            
            data = response.json()
            collections = [c['name'] for c in data.get('result', {}).get('collections', [])]
            
            success = True
            
            if 'faces_v1' not in collections:
                log_error("faces_v1 collection not found")
                success = False
            else:
                log_info("✓ faces_v1 collection verified")
            
            if 'identities_v1' not in collections:
                log_error("identities_v1 collection not found")
                success = False
            else:
                log_info("✓ identities_v1 collection verified")
            
            return success
            
        except Exception as e:
            log_error(f"Could not verify collections: {e}")
            return False
    
    def test_enrollment(self) -> bool:
        """Test enrollment endpoint."""
        log_info("Testing enrollment endpoint...")
        try:
            # Use minimal dummy data (will fail validation, but endpoint should respond)
            payload = {
                "tenant_id": "deployment-test",
                "identity_id": f"deploy-test-{int(time.time())}",
                "images_b64": ["data:image/jpeg;base64,/9j/4AAQSkZJRg=="]
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/enroll_identity",
                json=payload,
                timeout=self.timeout
            )
            
            # 200 = success, 422 = validation error (expected with dummy data)
            if response.status_code in [200, 422]:
                log_info(f"✓ Enrollment endpoint responds (HTTP {response.status_code})")
                return True
            else:
                log_error(f"Enrollment endpoint returned HTTP {response.status_code}")
                if response.text:
                    log_error(f"Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            log_error(f"Enrollment endpoint test failed: {e}")
            return False
    
    def test_verification(self) -> bool:
        """Test verification endpoint."""
        log_info("Testing verification endpoint...")
        try:
            # Use non-existent identity (will return 404, but endpoint should respond)
            payload = {
                "tenant_id": "deployment-test",
                "identity_id": f"non-existent-{int(time.time())}",
                "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/verify",
                json=payload,
                timeout=self.timeout
            )
            
            # 200 = success, 404 = not enrolled (expected), 422 = validation error (expected)
            if response.status_code in [200, 404, 422]:
                log_info(f"✓ Verification endpoint responds (HTTP {response.status_code})")
                return True
            else:
                log_error(f"Verification endpoint returned HTTP {response.status_code}")
                if response.text:
                    log_error(f"Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            log_error(f"Verification endpoint test failed: {e}")
            return False
    
    def test_search(self) -> bool:
        """Test search endpoint (legacy)."""
        log_info("Testing search endpoint (legacy)...")
        try:
            payload = {
                "tenant_id": "deployment-test",
                "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                "top_k": 10
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/search",
                json=payload,
                timeout=self.timeout
            )
            
            # 200 = success, 422 = validation error (expected with dummy data)
            if response.status_code in [200, 422]:
                log_info(f"✓ Search endpoint responds (HTTP {response.status_code})")
                return True
            else:
                log_error(f"Search endpoint returned HTTP {response.status_code}")
                if response.text:
                    log_error(f"Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            log_error(f"Search endpoint test failed: {e}")
            return False
    
    def run_smoke_tests(self) -> bool:
        """Run all smoke tests."""
        log_step("Running Smoke Tests")
        
        tests = [
            ("Enrollment", self.test_enrollment),
            ("Verification", self.test_verification),
            ("Search", self.test_search),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
                    self.errors.append(f"{name} test failed")
            except Exception as e:
                failed += 1
                self.errors.append(f"{name} test error: {e}")
        
        log_info(f"Smoke tests: {passed} passed, {failed} failed")
        
        if failed > 0:
            log_error("Some smoke tests failed")
            for error in self.errors:
                log_error(f"  - {error}")
            return False
        
        return True
    
    def run_pre_deployment_checks(self) -> bool:
        """Run pre-deployment checks."""
        log_step("Pre-Deployment Checks")
        
        checks = [
            ("Qdrant", self.check_qdrant),
            ("Collections", self.check_collections),
        ]
        
        all_passed = True
        for name, check_func in checks:
            if not check_func():
                all_passed = False
                self.errors.append(f"{name} check failed")
        
        return all_passed
    
    def run_post_deployment_verification(self) -> bool:
        """Run post-deployment verification."""
        log_step("Post-Deployment Verification")
        
        # Check API is up
        if not self.check_api():
            log_error("API not accessible after deployment")
            return False
        
        # Verify collections
        if not self.verify_collections_post_deploy():
            log_error("Collections verification failed")
            return False
        
        # Run smoke tests
        if not self.run_smoke_tests():
            return False
        
        return True

def main():
    """Main deployment flow."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [dev|staging|prod]")
        sys.exit(1)
    
    env = sys.argv[1].lower()
    
    if env == 'prod':
        print(f"\n{Colors.RED}WARNING: Deploying to PRODUCTION{Colors.NC}")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deployment cancelled.")
            sys.exit(0)
    
    print(f"\n{Colors.GREEN}{'='*50}{Colors.NC}")
    print(f"{Colors.GREEN}Verification-First Flow Deployment{Colors.NC}")
    print(f"{Colors.GREEN}{'='*50}{Colors.NC}")
    print(f"Environment: {Colors.BLUE}{env}{Colors.NC}")
    print(f"Qdrant URL: {ENV_CONFIGS[env]['qdrant_url']}")
    print(f"API URL: {ENV_CONFIGS[env]['api_url']}")
    print()
    
    checker = DeploymentChecker(env)
    
    # Pre-deployment checks
    if not checker.run_pre_deployment_checks():
        log_error("Pre-deployment checks failed")
        log_warn("Please fix issues and try again")
        sys.exit(1)
    
    # Deployment step (manual)
    log_step("Code Deployment")
    log_warn("Code deployment must be done manually")
    log_info("Options:")
    log_info("  - Docker: docker-compose up -d")
    log_info("  - Kubernetes: kubectl apply -f k8s/")
    log_info("  - Direct: systemctl restart face-pipeline")
    print()
    
    input("Press Enter after code deployment is complete...")
    
    # Post-deployment verification
    if not checker.run_post_deployment_verification():
        log_error("Post-deployment verification failed")
        log_warn("Please check logs and fix issues")
        sys.exit(1)
    
    # Success
    print()
    log_info(f"{Colors.GREEN}Deployment completed successfully!{Colors.NC}")
    print()
    log_info("Next steps:")
    log_info("  1. Monitor logs for any errors")
    log_info("  2. Check monitoring dashboards")
    log_info("  3. Run integration tests")
    log_info("  4. Verify with real user data (if safe)")
    print()

if __name__ == '__main__':
    main()


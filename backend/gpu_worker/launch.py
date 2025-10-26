"""
Windows GPU Worker Launcher

Robust launcher with process management, auto-recovery, and dependency checking.
Uses Windows mutex for single-instance enforcement and watchdog monitoring.
"""

import os
import sys
import logging
import subprocess
import time
import socket
import tempfile
import threading
import signal
import atexit
from pathlib import Path

# Try to import Windows-specific modules, but don't fail if they're not available
try:
    import win32api
    import win32event
    import win32process
    import win32con
    WINDOWS_MODULES_AVAILABLE = True
except ImportError:
    WINDOWS_MODULES_AVAILABLE = False
    logging.warning("Windows-specific modules not available. Some features may be limited.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Process management features may be limited.")

# Configure logging with file output
log_file = Path(__file__).parent / "gpu_worker.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Process management
_mutex = None
_watchdog_thread = None
_worker_process = None
_shutdown_event = threading.Event()

# Configuration
WORKER_PORT = 8765
HEALTH_CHECK_INTERVAL = 5.0  # Interval between health checks (not used currently)
WATCHDOG_INTERVAL = 15.0  # How often to check worker health (seconds)
MAX_RESTART_ATTEMPTS = 5  # Maximum restart attempts before giving up
RESTART_DELAY = 5.0  # Delay between restart attempts (seconds)
HEALTH_CHECK_TIMEOUT = 30.0  # Timeout for health check requests (seconds)
CONSECUTIVE_FAILURES_BEFORE_RESTART = 2  # Require multiple failures before restart

def list_existing_gpu_workers():
    """List existing GPU worker processes for visibility without killing them."""
    logger.info("Checking for existing GPU worker processes...")
    
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available, cannot check for existing processes")
        return True
    
    try:
        found_count = 0
        current_pid = os.getpid()
        
        # Find all Python processes that might be GPU workers
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Skip our own process
                if proc.info['pid'] == current_pid:
                    continue
                    
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline and any('worker.py' in arg for arg in cmdline):
                        # Only show actual worker.py processes, not launcher processes
                        if not any('launch.py' in arg for arg in cmdline):
                            logger.info(f"Found existing GPU worker process PID {proc.info['pid']}")
                            logger.info(f"Command: {' '.join(cmdline)}")
                            found_count += 1
                            
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process might have terminated or we don't have access
                continue
            except Exception as e:
                logger.debug(f"Error checking process {proc.info.get('pid', 'unknown')}: {e}")
                continue
        
        if found_count > 0:
            logger.info(f"Found {found_count} existing GPU worker process(es)")
        else:
            logger.info("No existing GPU worker processes found")
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking existing GPU workers: {e}")
        return False

def check_port_available(port=8765):
    """Check if the specified port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            logger.info(f"Port {port} is available")
            return True
    except OSError as e:
        logger.error(f"Port {port} is not available: {e}")
        return False

def acquire_process_lock():
    """Acquire a Windows mutex to prevent multiple instances."""
    global _mutex
    
    try:
        # Create a named mutex for single-instance enforcement
        mutex_name = "MordeauxGPUWorkerMutex"
        _mutex = win32event.CreateMutex(None, False, mutex_name)
        
        # Try to acquire the mutex (non-blocking)
        result = win32event.WaitForSingleObject(_mutex, 0)
        
        if result == win32event.WAIT_OBJECT_0:
            logger.info("Process lock acquired successfully")
            return True
        elif result == win32event.WAIT_ABANDONED:
            logger.warning("Previous instance was abandoned, taking over")
            return True
        else:
            logger.error("Another GPU worker instance is already running")
            return False
            
    except Exception as e:
        logger.error(f"Failed to acquire process lock: {e}")
        return False

def release_process_lock():
    """Release the process lock."""
    global _mutex
    
    try:
        if _mutex:
            win32event.ReleaseMutex(_mutex)
            win32api.CloseHandle(_mutex)
            _mutex = None
            logger.info("Process lock released")
    except Exception as e:
        logger.error(f"Failed to release process lock: {e}")

def check_directml_availability():
    """Check if DirectML is available."""
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        directml_available = 'DmlExecutionProvider' in available_providers
        
        logger.info(f"Available ONNX providers: {available_providers}")
        logger.info(f"DirectML available: {directml_available}")
        
        return directml_available
    except ImportError:
        logger.error("ONNX Runtime not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking DirectML availability: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'fastapi',
        'uvicorn',
        'onnxruntime',
        'insightface',
        'opencv-python',
        'numpy',
        'pillow',
        'psutil',
        'httpx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle special cases for import names
            import_name = package.replace('-', '_')
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'pillow':
                import_name = 'PIL'
            
            __import__(import_name)
            logger.info(f"[OK] {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"[MISSING] {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing GPU worker dependencies...")
    
    try:
        # Get the directory containing this script
        script_dir = Path(__file__).parent
        requirements_file = script_dir / "requirements.txt"
        
        if requirements_file.exists():
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            logger.info("Dependencies installed successfully")
            return True
        else:
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def get_worker_stats():
    """Get worker process statistics for monitoring."""
    global _worker_process
    
    if not _worker_process or not PSUTIL_AVAILABLE:
        return {}
    
    try:
        import psutil
        process = psutil.Process(_worker_process.pid)
        
        return {
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=0.1),
            'num_threads': process.num_threads(),
            'status': process.status()
        }
    except Exception as e:
        logger.debug(f"Failed to get worker stats: {e}")
        return {}

def check_worker_health():
    """Check if the worker process is healthy."""
    global _worker_process
    
    if not _worker_process:
        logger.warning("Health check failed: No worker process exists")
        return False
    
    try:
        # Check if process is still running
        poll_result = _worker_process.poll()
        if poll_result is not None:
            logger.error(f"Worker process has terminated with exit code: {poll_result}")
            # Try to get stderr/stdout if available
            try:
                if _worker_process.stderr:
                    stderr_output = _worker_process.stderr.read()
                    if stderr_output:
                        logger.error(f"Worker stderr: {stderr_output[:500]}")  # First 500 chars
            except:
                pass
            return False
        
        # Try to make a health check request
        import httpx
        logger.debug(f"Attempting health check to http://localhost:{WORKER_PORT}/health")
        
        try:
            with httpx.Client(timeout=HEALTH_CHECK_TIMEOUT) as client:
                response = client.get(f"http://localhost:{WORKER_PORT}/health")
                logger.debug(f"Health check response: status={response.status_code}")
                
                if response.status_code == 200:
                    health_data = response.json()
                    is_healthy = health_data.get("status") == "healthy"
                    
                    if is_healthy:
                        logger.debug(f"Health check passed - uptime: {health_data.get('uptime_seconds', 0):.1f}s")
                        # Add worker stats
                        stats = get_worker_stats()
                        if stats:
                            logger.debug(f"Worker stats: {stats['memory_mb']:.1f}MB RAM, {stats['cpu_percent']:.1f}% CPU, {stats['num_threads']} threads")
                    else:
                        logger.warning(f"Health check returned unhealthy status: {health_data}")
                    
                    return is_healthy
                else:
                    logger.error(f"Health check failed: HTTP {response.status_code}")
                    return False
                    
        except httpx.TimeoutException as e:
            logger.error(f"Health check timeout after {HEALTH_CHECK_TIMEOUT}s: {e}")
            return False
        except httpx.ConnectError as e:
            logger.error(f"Health check connection failed (worker may not be listening): {e}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Health check request error: {e}")
            return False
        
        return False
        
    except Exception as e:
        logger.error(f"Health check failed with unexpected error: {type(e).__name__}: {e}", exc_info=True)
        return False

def start_worker_process():
    """Start the GPU worker process."""
    global _worker_process
    
    try:
        # Get the worker script path
        script_dir = Path(__file__).parent
        worker_script = script_dir / "worker.py"
        
        if not worker_script.exists():
            logger.error(f"Worker script not found: {worker_script}")
            return False
        
        logger.info("Starting GPU worker process...")
        
        # Start the worker process
        _worker_process = subprocess.Popen([
            sys.executable, str(worker_script)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for startup
        time.sleep(2)
        
        # Check if process started successfully
        if _worker_process.poll() is not None:
            stdout, stderr = _worker_process.communicate()
            logger.error(f"Worker process failed to start: {stderr}")
            return False
        
        logger.info(f"GPU worker process started with PID {_worker_process.pid}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start worker process: {e}")
        return False

def stop_worker_process():
    """Stop the GPU worker process gracefully."""
    global _worker_process
    
    if _worker_process:
        try:
            logger.info("Stopping GPU worker process...")
            
            # Try graceful shutdown first
            _worker_process.terminate()
            
            # Wait for graceful shutdown
            try:
                _worker_process.wait(timeout=10)
                logger.info("Worker process stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Worker process did not stop gracefully, forcing termination")
                _worker_process.kill()
                _worker_process.wait()
            
            _worker_process = None
            
        except Exception as e:
            logger.error(f"Error stopping worker process: {e}")

def watchdog_monitor():
    """Watchdog thread to monitor worker health and restart if needed."""
    restart_attempts = 0
    consecutive_failures = 0
    last_health_check_time = time.time()
    
    while not _shutdown_event.is_set():
        try:
            current_time = time.time()
            time_since_last_check = current_time - last_health_check_time
            
            logger.debug(f"Performing health check ({time_since_last_check:.1f}s since last check)")
            
            if not check_worker_health():
                consecutive_failures += 1
                logger.warning(f"Worker health check failed (consecutive failures: {consecutive_failures})")
                
                # Only restart after 2 consecutive failures to avoid false positives
                if consecutive_failures >= CONSECUTIVE_FAILURES_BEFORE_RESTART:
                    if restart_attempts < MAX_RESTART_ATTEMPTS:
                        logger.info(f"Attempting to restart worker (attempt {restart_attempts + 1} after {consecutive_failures} failures)")
                        
                        # Stop current process
                        stop_worker_process()
                        
                        # Wait before restart
                        time.sleep(RESTART_DELAY)
                        
                        # Start new process
                        if start_worker_process():
                            restart_attempts = 0  # Reset on successful restart
                            consecutive_failures = 0  # Reset failure counter
                            logger.info("Worker restarted successfully")
                        else:
                            restart_attempts += 1
                            logger.error(f"Failed to restart worker (attempt {restart_attempts})")
                    else:
                        logger.error("Max restart attempts reached, giving up")
                        break
            else:
                # Reset consecutive failures on successful health check
                consecutive_failures = 0
                restart_attempts = 0  # Reset restart counter on successful health
            
            last_health_check_time = current_time
            
            # Wait before next check
            _shutdown_event.wait(WATCHDOG_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in watchdog monitor: {e}", exc_info=True)
            time.sleep(WATCHDOG_INTERVAL)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_event.set()
    
    # Stop worker process
    stop_worker_process()
    
    # Release process lock
    release_process_lock()
    
    logger.info("GPU Worker launcher shutdown complete")
    sys.exit(0)

def main():
    """Main launcher function."""
    logger.info("=== Windows GPU Worker Launcher v2.0 ===")
    
    # Check if we're on Windows
    if sys.platform != "win32":
        logger.warning("This launcher is designed for Windows. Current platform: " + sys.platform)
    
    # List existing GPU worker processes for visibility
    list_existing_gpu_workers()
    
    # Acquire process lock to prevent multiple instances
    if not acquire_process_lock():
        logger.error("Cannot start GPU worker: another instance is already running")
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(lambda: signal_handler(None, None))
    
    try:
        # Check if port is available
        if not check_port_available(WORKER_PORT):
            logger.error(f"Port {WORKER_PORT} is not available. Please stop any existing GPU worker.")
            sys.exit(1)
        
        # Check dependencies
        if not check_dependencies():
            logger.info("Attempting to install missing dependencies...")
            if not install_dependencies():
                logger.error("Failed to install dependencies. Please install manually.")
                sys.exit(1)
            
            # Check again after installation
            if not check_dependencies():
                logger.error("Dependencies still missing after installation attempt.")
                sys.exit(1)
        
        # Check DirectML availability
        directml_available = check_directml_availability()
        if not directml_available:
            logger.warning("DirectML not available. GPU worker will use CPU fallback.")
        else:
            logger.info("DirectML is available. GPU acceleration will be used.")
        
        # Start the worker process
        if not start_worker_process():
            logger.error("Failed to start worker process")
            sys.exit(1)
        
        # Start watchdog thread
        _watchdog_thread = threading.Thread(
            target=watchdog_monitor,
            name="watchdog",
            daemon=True
        )
        _watchdog_thread.start()
        logger.info("Watchdog thread started")
        
        # Wait for shutdown signal
        logger.info("GPU Worker launcher running. Press Ctrl+C to stop.")
        _shutdown_event.wait()
        
    except KeyboardInterrupt:
        logger.info("GPU Worker stopped by user")
    except Exception as e:
        logger.error(f"GPU Worker failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        _shutdown_event.set()
        stop_worker_process()
        release_process_lock()

if __name__ == "__main__":
    main()
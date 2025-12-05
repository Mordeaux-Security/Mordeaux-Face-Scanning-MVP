"""
Timing Logger for New Crawler System

Thread-safe singleton logger for detailed performance timing instrumentation.
Writes structured timing data to timings.txt for performance analysis.
"""

import threading
import time
from datetime import datetime
from typing import Optional
import os


class TimingLogger:
    """Thread-safe singleton timing logger."""
    
    _instance: Optional['TimingLogger'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._file_lock = threading.Lock()
        self._timing_file = "timings.txt"
        self._initialized = True
        
        # DISABLED: Debug file writing to reduce memory usage
        # Ensure timing file exists and is empty
        # with self._file_lock:
        #     with open(self._timing_file, 'w') as f:
        #         f.write("")  # Clear file
        pass
    
    def _log(self, event_type: str, *args):
        """Internal method to write timing log entry."""
        # DISABLED: Debug file writing to reduce memory usage
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # args_str = " | ".join(str(arg) for arg in args) if args else ""
        # log_line = f"[{timestamp}] {event_type}"
        # if args_str:
        #     log_line += f" | {args_str}"
        # 
        # with self._file_lock:
        #     with open(self._timing_file, 'a') as f:
        #         f.write(log_line + "\n")
        pass
    
    def log_system_start(self):
        """Log application launch."""
        self._log("SYSTEM_START")
    
    def log_crawl_start(self):
        """Log crawl begins."""
        self._log("CRAWL_START")
    
    def log_site_start(self, site_id: str, url: str):
        """Log site crawl begins."""
        self._log("SITE_START", site_id, url)
    
    def log_page_start(self, site_id: str, page_url: str):
        """Log page fetch begins."""
        self._log("PAGE_START", site_id, page_url)
    
    def log_page_end(self, site_id: str, page_url: str, duration_ms: float, candidate_count: int):
        """Log page fetch ends."""
        self._log("PAGE_END", site_id, page_url, f"{duration_ms:.0f}ms", f"{candidate_count} candidates")
    
    def log_site_end(self, site_id: str, duration_ms: float, total_pages: int, total_candidates: int):
        """Log site crawl ends."""
        self._log("SITE_END", site_id, f"{duration_ms:.0f}ms", f"{total_pages} pages", f"{total_candidates} candidates")
    
    def log_extraction_start(self, site_id: str, image_url: str):
        """Log image extraction begins."""
        self._log("EXTRACT_START", site_id, image_url)
    
    def log_extraction_end(self, site_id: str, image_url: str, duration_ms: float):
        """Log image extraction ends."""
        self._log("EXTRACT_END", site_id, image_url, f"{duration_ms:.0f}ms")
    
    def log_gpu_batch_start(self, batch_id: str, image_count: int):
        """Log GPU batch begins."""
        self._log("GPU_BATCH_START", batch_id, f"{image_count} images")
    
    def log_gpu_recognition_start(self, batch_id: str):
        """Log facial recognition begins."""
        self._log("GPU_RECOGNITION_START", batch_id)
    
    def log_gpu_recognition_end(self, batch_id: str, duration_ms: float, face_count: int):
        """Log facial recognition ends."""
        self._log("GPU_RECOGNITION_END", batch_id, f"{duration_ms:.0f}ms", f"{face_count} faces")
    
    def log_gpu_crop_start(self, image_id: str, face_index: int):
        """Log thumbnail crop begins."""
        self._log("GPU_CROP_START", image_id, f"face_{face_index}")
    
    def log_gpu_crop_end(self, image_id: str, face_index: int, duration_ms: float):
        """Log thumbnail crop ends."""
        self._log("GPU_CROP_END", image_id, f"face_{face_index}", f"{duration_ms:.0f}ms")
    
    def log_gpu_storage_start(self, image_id: str, storage_type: str):
        """Log storage operation begins (raw/thumb)."""
        self._log("GPU_STORAGE_START", image_id, storage_type)
    
    def log_gpu_storage_end(self, image_id: str, storage_type: str, duration_ms: float):
        """Log storage operation ends."""
        self._log("GPU_STORAGE_END", image_id, storage_type, f"{duration_ms:.0f}ms")
    
    def log_gpu_batch_end(self, batch_id: str, duration_ms: float, images_processed: int):
        """Log GPU batch ends."""
        self._log("GPU_BATCH_END", batch_id, f"{duration_ms:.0f}ms", f"{images_processed} images")
    
    def log_storage_start(self, site_id: str, image_id: str):
        """Log storage operation begins."""
        self._log("STORAGE_START", site_id, image_id)
    
    def log_storage_end(self, site_id: str, image_id: str, duration_ms: float, faces_count: int, raw_saved: bool, thumbs_count: int):
        """Log storage operation ends."""
        self._log("STORAGE_END", site_id, image_id, f"{duration_ms:.0f}ms", f"{faces_count} faces", f"raw={raw_saved}", f"{thumbs_count} thumbs")
    
    def log_crawl_end(self, duration_ms: float, total_sites: int, total_images: int):
        """Log crawl ends."""
        self._log("CRAWL_END", f"{duration_ms:.0f}ms", f"{total_sites} sites", f"{total_images} images")
    
    def log_system_shutdown(self):
        """Log application shutdown."""
        self._log("SYSTEM_SHUTDOWN")


def get_timing_logger() -> TimingLogger:
    """Get singleton timing logger instance."""
    return TimingLogger()

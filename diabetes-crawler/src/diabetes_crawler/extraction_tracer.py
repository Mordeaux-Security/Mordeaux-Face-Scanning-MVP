"""
Extraction Pipeline Tracer

Tracks every extraction attempt with structured logging to enable debugging
and analysis of extraction failures and strategy effectiveness.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ExtractionAttempt:
    """Structured data for a single extraction attempt."""
    url: str
    page_type: str
    strategy_used: Optional[str] = None
    success: bool = False
    failure_reason: Optional[str] = None
    content_length: int = 0
    title_found: bool = False
    author_found: bool = False
    date_found: bool = False
    timestamp: float = 0.0
    html_sample: Optional[str] = None
    strategy_order: List[str] = None
    strategy_results: Dict[str, Any] = None


class ExtractionTracer:
    """Tracks extraction attempts and writes structured logs."""
    
    def __init__(self):
        self.attempts: List[ExtractionAttempt] = []
        self.debug_dir = Path("crawl_output/debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def log_attempt(
        self,
        url: str,
        page_type: str,
        strategy_used: Optional[str] = None,
        success: bool = False,
        failure_reason: Optional[str] = None,
        content_length: int = 0,
        title_found: bool = False,
        author_found: bool = False,
        date_found: bool = False,
        html_sample: Optional[str] = None,
        strategy_order: Optional[List[str]] = None,
        strategy_results: Optional[Dict[str, Any]] = None
    ):
        """Log an extraction attempt."""
        attempt = ExtractionAttempt(
            url=url,
            page_type=page_type,
            strategy_used=strategy_used,
            success=success,
            failure_reason=failure_reason,
            content_length=content_length,
            title_found=title_found,
            author_found=author_found,
            date_found=date_found,
            timestamp=time.time(),
            html_sample=html_sample[:10000] if html_sample else None,  # Limit size
            strategy_order=strategy_order or [],
            strategy_results=strategy_results or {}
        )
        self.attempts.append(attempt)
        
    def flush(self, site_id: Optional[str] = None):
        """Write all attempts to file."""
        # DISABLED: Debug file writing to reduce memory usage
        # if not self.attempts:
        #     return
        #     
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"extraction_trace_{site_id}_{timestamp}.json" if site_id else f"extraction_trace_{timestamp}.json"
        # trace_file = self.debug_dir / filename
        # 
        # # Convert attempts to dict format
        # attempts_data = [asdict(attempt) for attempt in self.attempts]
        # 
        # with open(trace_file, 'w', encoding='utf-8') as f:
        #     json.dump({
        #         'timestamp': timestamp,
        #         'site_id': site_id,
        #         'total_attempts': len(attempts_data),
        #         'successful': sum(1 for a in attempts_data if a['success']),
        #         'failed': sum(1 for a in attempts_data if not a['success']),
        #         'attempts': attempts_data
        #     }, f, indent=2, default=str)
        # 
        # logger.info(f"[EXTRACTION-TRACER] Wrote {len(attempts_data)} extraction attempts to {trace_file}")
        self.attempts.clear()


# Singleton pattern
_tracer_instance = None

def get_extraction_tracer() -> ExtractionTracer:
    """Get or create extraction tracer instance."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = ExtractionTracer()
    return _tracer_instance


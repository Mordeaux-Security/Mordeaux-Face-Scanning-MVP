"""
JavaScript Rendering Service for Dynamic Content Crawling

This module provides JavaScript rendering capabilities using Playwright,
integrating seamlessly with the existing HTTP service and crawler architecture.
Maintains all existing optimizations while adding dynamic content support.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from urllib.parse import urlparse
import psutil
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from .config import CrawlerConfig
from .http_service import validate_url_security

logger = logging.getLogger(__name__)

@dataclass
class JSRenderingConfig:
    """Configuration for JavaScript rendering."""
    timeout: float = 30.0
    wait_time: float = 5.0
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent: str = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

@dataclass
class RenderedContent:
    """Container for rendered page content."""
    html: str
    final_url: str
    render_time: float
    memory_used: int
    images_loaded: int
    scripts_executed: int

class JSRenderingService:
    """
    JavaScript rendering service with resource management and optimization.
    
    Integrates with existing crawler architecture while maintaining:
    - Memory and CPU limits
    - Concurrency controls
    - Error handling and fallbacks
    - Caching capabilities
    """
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._contexts: Dict[str, BrowserContext] = {}
        self._active_sessions = 0
        self._max_concurrent = config.js_max_concurrent
        self._memory_limit = config.js_memory_limit
        self._cpu_limit = config.js_cpu_limit
        
        # Performance tracking
        self._total_renders = 0
        self._successful_renders = 0
        self._failed_renders = 0
        self._total_render_time = 0.0
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_playwright()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_playwright(self):
        """Ensure Playwright is initialized."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            logger.info("Playwright initialized for JavaScript rendering")
    
    async def _ensure_browser(self):
        """Ensure browser is launched with resource limits."""
        if self._browser is None or not self._browser.is_connected():
            # Ensure Playwright is initialized first
            await self._ensure_playwright()
            
            # Check system resources before launching browser
            if not self._check_system_resources():
                raise RuntimeError("Insufficient system resources for JavaScript rendering")
            
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.js_headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    f'--memory-pressure-off',
                    f'--max_old_space_size={self._memory_limit // (1024 * 1024)}',  # Convert to MB
                ]
            )
            logger.info("Browser launched for JavaScript rendering")
    
    def _check_system_resources(self) -> bool:
        """Check if system has sufficient resources for JavaScript rendering."""
        try:
            # Check memory usage - ensure we have enough available memory for JS rendering
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            required_memory_gb = self._memory_limit / (1024**3)
            
            # Dynamic memory limit adjustment based on available memory
            memory_percent = memory.percent
            if memory_percent > 85:  # Critical memory pressure
                # Reduce memory requirement by 50% for critical situations
                adjusted_required_gb = required_memory_gb * 0.5
                logger.info(f"Critical memory pressure ({memory_percent:.1f}%), reducing JS memory requirement from {required_memory_gb:.2f}GB to {adjusted_required_gb:.2f}GB")
                required_memory_gb = adjusted_required_gb
            elif memory_percent > 75:  # High memory pressure
                # Reduce memory requirement by 25% for high pressure
                adjusted_required_gb = required_memory_gb * 0.75
                logger.info(f"High memory pressure ({memory_percent:.1f}%), reducing JS memory requirement from {required_memory_gb:.2f}GB to {adjusted_required_gb:.2f}GB")
                required_memory_gb = adjusted_required_gb
            
            if available_memory_gb < required_memory_gb:
                logger.warning(f"Insufficient available memory for JS rendering: {available_memory_gb:.2f}GB available, {required_memory_gb:.2f}GB required")
                return False
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self._cpu_limit:
                logger.warning(f"CPU usage too high for JS rendering: {cpu_percent}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return False
    
    async def _cleanup_memory_before_rendering(self) -> None:
        """Perform proactive memory cleanup before JavaScript rendering."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Only perform cleanup if memory pressure is moderate or higher
            if memory_percent > 70:
                logger.info(f"Memory pressure detected ({memory_percent:.1f}%), performing cleanup before JS rendering")
                
                # Force garbage collection
                import gc
                collected = gc.collect()
                logger.info(f"Garbage collection freed {collected} objects")
                
                # Close unused browser contexts if memory is critical
                if memory_percent > 85 and len(self._contexts) > 1:
                    # Close oldest context (except default)
                    oldest_session = None
                    for session_id in list(self._contexts.keys()):
                        if session_id != "default":
                            oldest_session = session_id
                            break
                    
                    if oldest_session:
                        try:
                            await self._contexts[oldest_session].close()
                            del self._contexts[oldest_session]
                            logger.info(f"Closed unused browser context: {oldest_session}")
                        except Exception as e:
                            logger.warning(f"Error closing browser context {oldest_session}: {e}")
                
                # Small delay to allow memory cleanup to take effect
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.warning(f"Error during memory cleanup: {e}")
    
    async def _monitor_memory_during_rendering(self) -> None:
        """Monitor memory usage during JavaScript rendering and take action if needed."""
        try:
            while True:
                await asyncio.sleep(1)  # Check every second
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # If memory usage becomes critical during rendering, log warning
                if memory_percent > 90:
                    logger.warning(f"Critical memory usage during JS rendering: {memory_percent:.1f}%")
                    # Could implement emergency cleanup here if needed
                elif memory_percent > 85:
                    logger.info(f"High memory usage during JS rendering: {memory_percent:.1f}%")
                    
        except asyncio.CancelledError:
            # Task was cancelled, which is expected when rendering completes
            pass
        except Exception as e:
            logger.warning(f"Error in memory monitoring: {e}")
    
    async def _get_or_create_context(self, session_id: str) -> BrowserContext:
        """Get or create a browser context for a session."""
        if session_id not in self._contexts:
            await self._ensure_browser()
            
            self._contexts[session_id] = await self._browser.new_context(
                viewport={
                    'width': self.config.js_viewport_width,
                    'height': self.config.js_viewport_height
                },
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                java_script_enabled=True,
                ignore_https_errors=True,
                bypass_csp=True
            )
            
            # Set resource limits for this context
            await self._contexts[session_id].set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            })
        
        return self._contexts[session_id]
    
    async def render_page(
        self, 
        url: str, 
        config: Optional[JSRenderingConfig] = None,
        session_id: Optional[str] = None
    ) -> Tuple[Optional[RenderedContent], str]:
        """
        Render a page with JavaScript execution.
        
        Args:
            url: URL to render
            config: Rendering configuration
            session_id: Session identifier for context reuse
            
        Returns:
            Tuple of (rendered_content, status_code)
        """
        if not self.config.js_enabled:
            return None, "JS_RENDERING_DISABLED"
        
        if config is None:
            config = JSRenderingConfig()
        
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        # Check concurrency limits
        if self._active_sessions >= self._max_concurrent:
            return None, "MAX_CONCURRENT_EXCEEDED"
        
        # Validate URL security
        is_safe, reason = validate_url_security(url)
        if not is_safe:
            logger.warning(f"JS rendering URL rejected: {reason} - {url}")
            return None, f"BLOCKED_{reason}"
        
        start_time = time.time()
        self._active_sessions += 1
        self._total_renders += 1
        
        try:
            # Proactive memory cleanup before rendering
            await self._cleanup_memory_before_rendering()
            
            # Check system resources again before rendering
            if not self._check_system_resources():
                return None, "INSUFFICIENT_RESOURCES"
            
            # Get or create browser context
            context = await self._get_or_create_context(session_id)
            
            # Create new page
            page = await context.new_page()
            
            try:
                # Set timeout for page operations
                page.set_default_timeout(config.timeout * 1000)  # Convert to milliseconds
                
                # Monitor memory during rendering
                memory_monitor_task = asyncio.create_task(self._monitor_memory_during_rendering())
                
                # Navigate to the page
                response = await page.goto(url, wait_until='domcontentloaded')
                
                if not response or response.status >= 400:
                    return None, f"HTTP_ERROR_{response.status if response else 'NO_RESPONSE'}"
                
                # Wait for page to stabilize
                await asyncio.sleep(config.wait_time)
                
                # Wait for DOM content to load
                try:
                    await page.wait_for_load_state('domcontentloaded', timeout=5000)
                except:
                    # Continue even if timeout - we have the content we need
                    pass
                
                # Wait for images to load (especially lazy-loaded ones)
                try:
                    # Wait for at least one image to be visible
                    await page.wait_for_selector('img', timeout=3000)
                    
                    # Trigger lazy loading by scrolling
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(1)
                    await page.evaluate("window.scrollTo(0, 0)")
                    await asyncio.sleep(1)
                    
                    # Wait for network to be idle (images loading)
                    try:
                        await page.wait_for_load_state('networkidle', timeout=3000)
                    except:
                        # Continue if network doesn't become idle
                        pass
                        
                except:
                    # Continue even if no images found
                    pass
                
                # Get final URL after redirects
                final_url = page.url
                
                # Get rendered HTML
                html_content = await page.content()
                
                # Get performance metrics
                render_time = time.time() - start_time
                memory_used = await self._get_page_memory_usage(page)
                
                # Count loaded images and executed scripts
                images_loaded = await page.evaluate("document.images.length")
                scripts_executed = await page.evaluate("document.scripts.length")
                
                # Create rendered content object
                rendered_content = RenderedContent(
                    html=html_content,
                    final_url=final_url,
                    render_time=render_time,
                    memory_used=memory_used,
                    images_loaded=images_loaded,
                    scripts_executed=scripts_executed
                )
                
                self._successful_renders += 1
                self._total_render_time += render_time
                
                logger.info(f"Successfully rendered {url} in {render_time:.2f}s")
                return rendered_content, "200"
                
            finally:
                # Cancel memory monitoring task
                if 'memory_monitor_task' in locals():
                    memory_monitor_task.cancel()
                    try:
                        await memory_monitor_task
                    except asyncio.CancelledError:
                        pass
                
                # Always close the page
                await page.close()
                
        except Exception as e:
            self._failed_renders += 1
            error_msg = f"JavaScript rendering failed for {url}: {str(e)}"
            logger.error(error_msg)
            return None, f"RENDER_ERROR_{str(e)}"
            
        finally:
            self._active_sessions -= 1
    
    async def _get_page_memory_usage(self, page: Page) -> int:
        """Get memory usage of a page."""
        try:
            # Get memory usage from browser metrics
            metrics = await page.evaluate("""
                () => {
                    if (performance.memory) {
                        return performance.memory.usedJSHeapSize;
                    }
                    return 0;
                }
            """)
            return int(metrics) if metrics else 0
        except:
            return 0
    
    def detect_javascript_usage(self, html_content: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if a page likely requires JavaScript rendering.
        
        Args:
            html_content: Static HTML content
            
        Returns:
            Tuple of (needs_js_rendering, detection_info)
        """
        if not self.config.js_detection_enabled:
            return False, {"reason": "detection_disabled"}
        
        detection_info = {
            "script_count": 0,
            "js_keywords_found": [],
            "has_spa_indicators": False,
            "has_lazy_loading": False,
            "has_dynamic_content": False
        }
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Count script tags
            scripts = soup.find_all('script')
            detection_info["script_count"] = len(scripts)
            
            # Check for JavaScript keywords in content
            content_lower = html_content.lower()
            for keyword in self.config.js_detection_keywords:
                if keyword in content_lower:
                    detection_info["js_keywords_found"].append(keyword)
            
            # Check for SPA indicators
            spa_indicators = [
                'data-reactroot', 'ng-app', 'vue-app', 'id="app"',
                'class="app"', 'single-page', 'spa'
            ]
            for indicator in spa_indicators:
                if indicator in content_lower:
                    detection_info["has_spa_indicators"] = True
                    break
            
            # Check for lazy loading indicators
            lazy_indicators = [
                'lazy', 'lazyload', 'data-src', 'data-lazy',
                'loading="lazy"', 'infinite-scroll'
            ]
            for indicator in lazy_indicators:
                if indicator in content_lower:
                    detection_info["has_lazy_loading"] = True
                    break
            
            # Check for dynamic content indicators
            dynamic_indicators = [
                'dynamic', 'ajax', 'fetch', 'xhr', 'async',
                'onclick', 'onload', 'addEventListener'
            ]
            for indicator in dynamic_indicators:
                if indicator in content_lower:
                    detection_info["has_dynamic_content"] = True
                    break
            
            # Determine if JavaScript rendering is needed
            needs_js = (
                detection_info["script_count"] >= self.config.js_detection_script_threshold or
                len(detection_info["js_keywords_found"]) > 0 or
                detection_info["has_spa_indicators"] or
                detection_info["has_lazy_loading"] or
                detection_info["has_dynamic_content"]
            )
            
            return needs_js, detection_info
            
        except Exception as e:
            logger.error(f"Error detecting JavaScript usage: {e}")
            return False, {"error": str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the rendering service."""
        avg_render_time = (
            self._total_render_time / self._successful_renders 
            if self._successful_renders > 0 else 0
        )
        
        success_rate = (
            self._successful_renders / self._total_renders * 100 
            if self._total_renders > 0 else 0
        )
        
        return {
            "total_renders": self._total_renders,
            "successful_renders": self._successful_renders,
            "failed_renders": self._failed_renders,
            "success_rate": success_rate,
            "average_render_time": avg_render_time,
            "active_sessions": self._active_sessions,
            "max_concurrent": self._max_concurrent,
            "contexts_created": len(self._contexts)
        }
    
    async def close(self):
        """Close all browser contexts and cleanup resources."""
        try:
            # Close all contexts
            for context in self._contexts.values():
                await context.close()
            self._contexts.clear()
            
            # Close browser
            if self._browser:
                await self._browser.close()
                self._browser = None
            
            # Stop playwright
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            
            logger.info("JavaScript rendering service closed")
            
        except Exception as e:
            logger.error(f"Error closing JavaScript rendering service: {e}")

# Global JavaScript rendering service instance
_js_rendering_service: Optional[JSRenderingService] = None

async def get_js_rendering_service(config: CrawlerConfig) -> JSRenderingService:
    """Get the global JavaScript rendering service instance."""
    global _js_rendering_service
    if _js_rendering_service is None:
        _js_rendering_service = JSRenderingService(config)
    return _js_rendering_service

async def close_js_rendering_service():
    """Close the global JavaScript rendering service."""
    global _js_rendering_service
    if _js_rendering_service is not None:
        await _js_rendering_service.close()
        _js_rendering_service = None

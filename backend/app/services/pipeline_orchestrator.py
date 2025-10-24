"""
Pipeline Orchestrator

Coordinates all pipeline stages and maintains work queues between stages.
Balances producer/consumer rates and prevents pipeline stalls.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import threading

from .crawler_modules.resources import ResourceMonitor
from .adaptive_resource_manager import AdaptiveResourceManager, get_adaptive_resource_manager

logger = logging.getLogger(__name__)


@dataclass
class QueueMetrics:
    """Metrics for a pipeline queue."""
    current_depth: int = 0
    max_depth: int = 0
    total_processed: int = 0
    avg_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    last_processed_time: float = 0.0


class PipelineQueue:
    """
    A pipeline queue with backpressure and dynamic sizing.
    
    Automatically adjusts queue size based on throughput and latency.
    """
    
    def __init__(
        self,
        name: str,
        max_size: int = 100,
        min_size: int = 10,
        backpressure_threshold: float = 0.8
    ):
        """
        Initialize pipeline queue.
        
        Args:
            name: Queue name for identification
            max_size: Maximum queue size
            min_size: Minimum queue size
            backpressure_threshold: Threshold for applying backpressure (0-1)
        """
        self.name = name
        self.max_size = max_size
        self.min_size = min_size
        self.backpressure_threshold = backpressure_threshold
        
        # Queue state
        self._queue = asyncio.Queue(maxsize=max_size)
        self._metrics = QueueMetrics()
        self._processing_times = deque(maxlen=100)
        self._last_throughput_calc = time.time()
        self._throughput_samples = deque(maxlen=10)
        
        # Backpressure state
        self._backpressure_active = False
        self._backpressure_start_time = 0
        
        # Dynamic sizing
        self._target_size = max_size
        self._last_size_adjustment = time.time()
        self._size_adjustment_interval = 5.0  # Adjust size every 5 seconds
    
    async def put(self, item: Any) -> bool:
        """
        Put item in queue with backpressure handling.
        
        Args:
            item: Item to add to queue
            
        Returns:
            True if item was added, False if backpressure applied
        """
        try:
            # Check for backpressure
            if self._should_apply_backpressure():
                self._backpressure_active = True
                self._backpressure_start_time = time.time()
                logger.debug(f"Backpressure applied to queue {self.name}")
                return False
            
            # Add item to queue
            await self._queue.put(item)
            self._metrics.current_depth = self._queue.qsize()
            self._metrics.max_depth = max(self._metrics.max_depth, self._metrics.current_depth)
            
            # Reset backpressure if queue is not full
            if self._metrics.current_depth < self.max_size * self.backpressure_threshold:
                self._backpressure_active = False
            
            return True
            
        except asyncio.QueueFull:
            logger.warning(f"Queue {self.name} is full, applying backpressure")
            self._backpressure_active = True
            return False
    
    async def get(self) -> Any:
        """
        Get item from queue.
        
        Returns:
            Item from queue
        """
        start_time = time.time()
        item = await self._queue.get()
        
        # Update metrics
        processing_time = time.time() - start_time
        self._processing_times.append(processing_time)
        self._metrics.current_depth = self._queue.qsize()
        self._metrics.total_processed += 1
        self._metrics.last_processed_time = time.time()
        
        # Update throughput
        self._update_throughput()
        
        return item
    
    def _should_apply_backpressure(self) -> bool:
        """Check if backpressure should be applied."""
        if self._backpressure_active:
            # Continue backpressure for at least 1 second
            return time.time() - self._backpressure_start_time < 1.0
        
        # Apply backpressure if queue is nearly full
        utilization = self._metrics.current_depth / self.max_size
        return utilization > self.backpressure_threshold
    
    def _update_throughput(self):
        """Update throughput metrics."""
        current_time = time.time()
        
        # Calculate throughput over last 10 seconds
        if current_time - self._last_throughput_calc > 1.0:  # Update every second
            if self._processing_times:
                self._metrics.avg_processing_time = sum(self._processing_times) / len(self._processing_times)
            
            # Calculate items per second
            time_window = min(10.0, current_time - self._last_throughput_calc)
            items_processed = self._metrics.total_processed
            self._metrics.throughput_per_second = items_processed / time_window if time_window > 0 else 0
            
            self._throughput_samples.append(self._metrics.throughput_per_second)
            self._last_throughput_calc = current_time
    
    def get_metrics(self) -> QueueMetrics:
        """Get current queue metrics."""
        return self._metrics
    
    def is_healthy(self) -> bool:
        """Check if queue is healthy (not stalled)."""
        if self._metrics.total_processed == 0:
            return True  # No items processed yet
        
        # Check if queue has been stalled for too long
        time_since_last_processed = time.time() - self._metrics.last_processed_time
        return time_since_last_processed < 30.0  # Consider stalled if no processing for 30 seconds
    
    def adjust_size(self, new_size: int):
        """Adjust queue size dynamically."""
        new_size = max(self.min_size, min(self.max_size, new_size))
        
        if new_size != self._target_size:
            self._target_size = new_size
            self._last_size_adjustment = time.time()
            logger.debug(f"Queue {self.name} size adjusted to {new_size}")
    
    def task_done(self):
        """Mark task as done."""
        self._queue.task_done()


class PipelineOrchestrator:
    """
    Orchestrates pipeline stages and manages work queues.
    
    Coordinates all pipeline stages, maintains work queues between stages,
    and balances producer/consumer rates to prevent stalls.
    """
    
    def __init__(
        self,
        resource_manager: Optional[AdaptiveResourceManager] = None,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            resource_manager: AdaptiveResourceManager instance
            resource_monitor: ResourceMonitor instance
        """
        self.resource_manager = resource_manager or get_adaptive_resource_manager()
        self.resource_monitor = resource_monitor or self.resource_manager.resource_monitor
        
        # Pipeline queues
        self.queues: Dict[str, PipelineQueue] = {}
        
        # Pipeline stages
        self.stages: Dict[str, Dict[str, Any]] = {}
        
        # Coordination
        self._orchestration_lock = asyncio.Lock()
        self._health_check_interval = 5.0
        self._last_health_check = 0
        
        # Performance tracking
        self._stage_metrics = {}
        self._pipeline_throughput = 0.0
        
        # Initialize default queues
        self._initialize_default_queues()
    
    def _initialize_default_queues(self):
        """Initialize default pipeline queues."""
        default_queues = {
            'download': {'max_size': 100, 'min_size': 10},
            'process': {'max_size': 50, 'min_size': 5},
            'storage': {'max_size': 200, 'min_size': 20},
            'face_detection': {'max_size': 30, 'min_size': 3},
            'quality_check': {'max_size': 40, 'min_size': 4}
        }
        
        for name, config in default_queues.items():
            self.queues[name] = PipelineQueue(
                name=name,
                max_size=config['max_size'],
                min_size=config['min_size']
            )
    
    def register_stage(
        self,
        name: str,
        input_queue: str,
        output_queue: str,
        processor: Callable,
        concurrency: int = 1
    ):
        """
        Register a pipeline stage.
        
        Args:
            name: Stage name
            input_queue: Input queue name
            output_queue: Output queue name
            processor: Processing function
            concurrency: Number of concurrent workers
        """
        self.stages[name] = {
            'input_queue': input_queue,
            'output_queue': output_queue,
            'processor': processor,
            'concurrency': concurrency,
            'active_workers': 0,
            'total_processed': 0,
            'last_processed': 0
        }
        logger.info(f"Registered pipeline stage: {name}")
    
    async def start_stage(self, stage_name: str):
        """Start a pipeline stage."""
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        stage = self.stages[stage_name]
        input_queue = self.queues[stage['input_queue']]
        output_queue = self.queues[stage['output_queue']] if stage['output_queue'] else None
        
        # Start concurrent workers
        tasks = []
        for i in range(stage['concurrency']):
            task = asyncio.create_task(
                self._stage_worker(stage_name, input_queue, output_queue, i)
            )
            tasks.append(task)
        
        logger.info(f"Started {stage['concurrency']} workers for stage {stage_name}")
        return tasks
    
    async def _stage_worker(
        self,
        stage_name: str,
        input_queue: PipelineQueue,
        output_queue: Optional[PipelineQueue],
        worker_id: int
    ):
        """Worker for a pipeline stage."""
        stage = self.stages[stage_name]
        processor = stage['processor']
        
        logger.debug(f"Started worker {worker_id} for stage {stage_name}")
        
        try:
            while True:
                try:
                    # Get item from input queue
                    item = await input_queue.get()
                    
                    # Process item
                    start_time = time.time()
                    result = await processor(item)
                    processing_time = time.time() - start_time
                    
                    # Update stage metrics
                    stage['active_workers'] += 1
                    stage['total_processed'] += 1
                    stage['last_processed'] = time.time()
                    
                    # Put result in output queue if available
                    if output_queue and result is not None:
                        await output_queue.put(result)
                    
                    # Update resource monitor
                    self.resource_monitor.update_processing_latency(processing_time * 1000)  # Convert to ms
                    
                    # Mark task as done
                    input_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Error in stage {stage_name} worker {worker_id}: {e}")
                    input_queue.task_done()
                    
        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_id} for stage {stage_name} cancelled")
        finally:
            stage['active_workers'] -= 1
    
    async def orchestrate_pipeline(self):
        """Main orchestration loop."""
        logger.info("Starting pipeline orchestration")
        
        try:
            while True:
                await self._health_check()
                await self._adjust_queue_sizes()
                await self._balance_workload()
                await asyncio.sleep(self._health_check_interval)
                
        except asyncio.CancelledError:
            logger.info("Pipeline orchestration cancelled")
    
    async def _health_check(self):
        """Check pipeline health and identify issues."""
        current_time = time.time()
        
        if current_time - self._last_health_check < self._health_check_interval:
            return
        
        unhealthy_queues = []
        stalled_stages = []
        
        # Check queue health
        for name, queue in self.queues.items():
            if not queue.is_healthy():
                unhealthy_queues.append(name)
                logger.warning(f"Queue {name} appears to be stalled")
        
        # Check stage health
        for name, stage in self.stages.items():
            if stage['total_processed'] > 0:
                time_since_last = current_time - stage['last_processed']
                if time_since_last > 60:  # No processing for 1 minute
                    stalled_stages.append(name)
                    logger.warning(f"Stage {name} appears to be stalled")
        
        # Log health status
        if unhealthy_queues or stalled_stages:
            logger.warning(f"Pipeline health issues - Queues: {unhealthy_queues}, Stages: {stalled_stages}")
        else:
            logger.debug("Pipeline health check passed")
        
        self._last_health_check = current_time
    
    async def _adjust_queue_sizes(self):
        """Dynamically adjust queue sizes based on performance."""
        current_time = time.time()
        
        for name, queue in self.queues.items():
            # Only adjust if enough time has passed
            if current_time - queue._last_size_adjustment < queue._size_adjustment_interval:
                continue
            
            metrics = queue.get_metrics()
            
            # Calculate optimal size based on throughput and latency
            if metrics.throughput_per_second > 0:
                # Size = throughput * latency * safety_factor
                latency_seconds = metrics.avg_processing_time
                safety_factor = 2.0
                optimal_size = int(metrics.throughput_per_second * latency_seconds * safety_factor)
                optimal_size = max(queue.min_size, min(queue.max_size, optimal_size))
                
                queue.adjust_size(optimal_size)
    
    async def _balance_workload(self):
        """Balance workload across pipeline stages."""
        # Get current resource utilization
        resource_summary = self.resource_monitor.get_resource_summary()
        bottleneck = self.resource_monitor.identify_bottleneck()
        
        # Adjust stage concurrency based on bottleneck
        for name, stage in self.stages.items():
            if bottleneck == 'cpu' and 'process' in name:
                # Increase processing concurrency
                new_concurrency = min(stage['concurrency'] + 1, 16)
                if new_concurrency != stage['concurrency']:
                    stage['concurrency'] = new_concurrency
                    logger.debug(f"Increased concurrency for {name} to {new_concurrency}")
            
            elif bottleneck == 'io' and 'download' in name:
                # Increase download concurrency
                new_concurrency = min(stage['concurrency'] + 1, 32)
                if new_concurrency != stage['concurrency']:
                    stage['concurrency'] = new_concurrency
                    logger.debug(f"Increased concurrency for {name} to {new_concurrency}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        queue_status = {}
        for name, queue in self.queues.items():
            metrics = queue.get_metrics()
            queue_status[name] = {
                'current_depth': metrics.current_depth,
                'max_depth': metrics.max_depth,
                'total_processed': metrics.total_processed,
                'throughput_per_second': metrics.throughput_per_second,
                'avg_processing_time': metrics.avg_processing_time,
                'is_healthy': queue.is_healthy()
            }
        
        stage_status = {}
        for name, stage in self.stages.items():
            stage_status[name] = {
                'active_workers': stage['active_workers'],
                'total_processed': stage['total_processed'],
                'last_processed': stage['last_processed'],
                'concurrency': stage['concurrency']
            }
        
        return {
            'queues': queue_status,
            'stages': stage_status,
            'pipeline_throughput': self._pipeline_throughput,
            'resource_utilization': self.resource_monitor.get_resource_summary()
        }
    
    async def shutdown(self):
        """Shutdown pipeline orchestrator."""
        logger.info("Shutting down pipeline orchestrator")
        
        # Cancel all stage workers
        for stage_name in self.stages:
            # This would need to be implemented based on how workers are tracked
            pass
        
        # Clear queues
        for queue in self.queues.values():
            while not queue._queue.empty():
                try:
                    queue._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break


# Global instance
_pipeline_orchestrator: Optional[PipelineOrchestrator] = None


def get_pipeline_orchestrator() -> PipelineOrchestrator:
    """Get the global pipeline orchestrator instance."""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is None:
        _pipeline_orchestrator = PipelineOrchestrator()
    return _pipeline_orchestrator


def cleanup_pipeline_orchestrator():
    """Clean up the global pipeline orchestrator."""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is not None:
        asyncio.create_task(_pipeline_orchestrator.shutdown())
        _pipeline_orchestrator = None

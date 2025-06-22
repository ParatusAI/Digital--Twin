#!/usr/bin/env python3
"""
Real-Time Monitoring and Logging System for CsPbBr3 Digital Twin
Implements comprehensive logging, progress tracking, and performance monitoring
"""

import logging
import time
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from functools import wraps
import sys
import traceback

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.dates import DateFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    process_time: float = 0.0
    wall_time: float = 0.0
    samples_per_second: float = 0.0
    total_samples: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressState:
    """Container for progress tracking"""
    operation: str = ""
    current: int = 0
    total: int = 0
    percentage: float = 0.0
    eta_seconds: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    rate: float = 0.0
    status: str = "running"  # running, completed, failed, paused


class StructuredLogger:
    """Enhanced logger with structured logging capabilities"""
    
    def __init__(self, name: str = "cspbbr3_digital_twin", 
                 log_dir: Optional[str] = None,
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_json: Enable JSON structured logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup log directory
        if log_dir is None:
            log_dir = Path("logs")
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(exist_ok=True)
        self.log_dir = log_dir
        
        # Setup formatters
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if enable_file:
            file_path = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # JSON handler for structured logs
        if enable_json:
            json_path = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.json"
            json_handler = logging.FileHandler(json_path)
            json_handler.setFormatter(self._json_formatter())
            self.logger.addHandler(json_handler)
        
        # Track metrics
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        
    def _json_formatter(self):
        """Create JSON formatter for structured logging"""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_entry['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': traceback.format_exception(*record.exc_info)
                    }
                
                # Add extra fields from record
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                                 'filename', 'module', 'lineno', 'funcName', 'created', 
                                 'msecs', 'relativeCreated', 'thread', 'threadName', 
                                 'processName', 'process', 'message', 'exc_info', 'exc_text', 'stack_info']:
                        log_entry[key] = value
                
                return json.dumps(log_entry)
        
        return JSONFormatter()
    
    def log_with_metrics(self, level: str, message: str, 
                        custom_metrics: Optional[Dict[str, Any]] = None, **kwargs):
        """Log message with performance metrics"""
        # Collect system metrics
        metrics = self._collect_metrics(custom_metrics)
        
        # Add metrics to log record
        extra_data = {
            'metrics': asdict(metrics),
            **kwargs
        }
        
        # Log the message
        log_func = getattr(self.logger, level.lower())
        log_func(message, extra=extra_data)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 1000 entries)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _collect_metrics(self, custom_metrics: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """Collect current performance metrics"""
        if PSUTIL_AVAILABLE and psutil:
            try:
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
                memory_percent = process.memory_percent()
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu_percent = 0.0
                memory_mb = 0.0
                memory_percent = 0.0
        else:
            cpu_percent = 0.0
            memory_mb = 0.0
            memory_percent = 0.0
        
        wall_time = time.time() - self.start_time
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            wall_time=wall_time,
            custom_metrics=custom_metrics or {}
        )
    
    def info(self, message: str, **kwargs):
        """Log info message with metrics"""
        self.log_with_metrics("info", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with metrics"""
        self.log_with_metrics("debug", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with metrics"""
        self.log_with_metrics("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with metrics"""
        self.log_with_metrics("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with metrics"""
        self.log_with_metrics("critical", message, **kwargs)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        if not self.metrics_history:
            return {"status": "No metrics collected"}
        
        # Extract numeric metrics
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_mb for m in self.metrics_history]
        
        return {
            "collection_period": {
                "start": self.metrics_history[0].timestamp,
                "end": self.metrics_history[-1].timestamp,
                "duration_minutes": (time.time() - self.start_time) / 60
            },
            "cpu_usage": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0
            },
            "memory_usage": {
                "current_mb": memory_values[-1] if memory_values else 0,
                "average_mb": sum(memory_values) / len(memory_values) if memory_values else 0,
                "peak_mb": max(memory_values) if memory_values else 0,
                "min_mb": min(memory_values) if memory_values else 0
            },
            "total_metrics_collected": len(self.metrics_history)
        }


class ProgressTracker:
    """Advanced progress tracking with ETA and rate calculation"""
    
    def __init__(self, logger: Optional[StructuredLogger] = None):
        """Initialize progress tracker"""
        self.logger = logger
        self.active_operations: Dict[str, ProgressState] = {}
        self.completed_operations: List[ProgressState] = []
        self._lock = threading.Lock()
        
    def start_operation(self, operation_id: str, operation_name: str, 
                       total_items: int) -> str:
        """Start tracking a new operation"""
        with self._lock:
            progress_state = ProgressState(
                operation=operation_name,
                total=total_items,
                start_time=time.time()
            )
            
            self.active_operations[operation_id] = progress_state
            
            if self.logger:
                self.logger.info(f"Started operation: {operation_name}", 
                               operation_id=operation_id, total_items=total_items)
            
            return operation_id
    
    def update_progress(self, operation_id: str, current: int, 
                       status_message: Optional[str] = None):
        """Update progress for an operation"""
        with self._lock:
            if operation_id not in self.active_operations:
                return
            
            state = self.active_operations[operation_id]
            state.current = current
            state.last_update = time.time()
            
            # Calculate percentage
            if state.total > 0:
                state.percentage = (current / state.total) * 100
            
            # Calculate rate and ETA
            elapsed = state.last_update - state.start_time
            if elapsed > 0:
                state.rate = current / elapsed
                if state.rate > 0 and current < state.total:
                    remaining_items = state.total - current
                    state.eta_seconds = remaining_items / state.rate
            
            if status_message and self.logger:
                self.logger.debug(f"Progress update: {state.operation}", 
                                operation_id=operation_id,
                                current=current,
                                total=state.total,
                                percentage=state.percentage,
                                rate=state.rate,
                                eta_seconds=state.eta_seconds,
                                status=status_message)
    
    def complete_operation(self, operation_id: str, status: str = "completed"):
        """Mark operation as completed"""
        with self._lock:
            if operation_id not in self.active_operations:
                return
            
            state = self.active_operations[operation_id]
            state.status = status
            state.last_update = time.time()
            
            # Move to completed operations
            self.completed_operations.append(state)
            del self.active_operations[operation_id]
            
            # Calculate final metrics
            total_time = state.last_update - state.start_time
            final_rate = state.current / total_time if total_time > 0 else 0
            
            if self.logger:
                self.logger.info(f"Completed operation: {state.operation}",
                               operation_id=operation_id,
                               status=status,
                               total_time_seconds=total_time,
                               final_rate=final_rate,
                               items_completed=state.current)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of all progress tracking"""
        with self._lock:
            summary = {
                "active_operations": len(self.active_operations),
                "completed_operations": len(self.completed_operations),
                "active_details": {},
                "completed_summary": {}
            }
            
            # Active operations details
            for op_id, state in self.active_operations.items():
                summary["active_details"][op_id] = {
                    "operation": state.operation,
                    "progress_percent": state.percentage,
                    "current": state.current,
                    "total": state.total,
                    "rate": state.rate,
                    "eta_seconds": state.eta_seconds,
                    "elapsed_seconds": time.time() - state.start_time
                }
            
            # Completed operations summary
            if self.completed_operations:
                total_completed_items = sum(op.current for op in self.completed_operations)
                avg_completion_time = sum(op.last_update - op.start_time for op in self.completed_operations) / len(self.completed_operations)
                
                summary["completed_summary"] = {
                    "total_items_processed": total_completed_items,
                    "average_completion_time": avg_completion_time,
                    "operations_by_status": {}
                }
                
                # Group by status
                for op in self.completed_operations:
                    status = op.status
                    if status not in summary["completed_summary"]["operations_by_status"]:
                        summary["completed_summary"]["operations_by_status"][status] = 0
                    summary["completed_summary"]["operations_by_status"][status] += 1
            
            return summary


class EnhancedProgressBar:
    """Enhanced progress bar with rich information display"""
    
    def __init__(self, total: int, description: str = "", 
                 show_rate: bool = True, show_eta: bool = True,
                 update_interval: float = 0.1):
        """Initialize enhanced progress bar"""
        self.total = total
        self.description = description
        self.show_rate = show_rate
        self.show_eta = show_eta
        self.update_interval = update_interval
        
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        if TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=total,
                desc=description,
                unit="samples",
                unit_scale=True,
                dynamic_ncols=True,
                miniters=1
            )
        else:
            self.pbar = None
            self._manual_progress_display()
    
    def update(self, n: int = 1, postfix: Optional[Dict[str, Any]] = None):
        """Update progress bar"""
        self.current += n
        current_time = time.time()
        
        if TQDM_AVAILABLE and self.pbar:
            if postfix:
                self.pbar.set_postfix(postfix)
            self.pbar.update(n)
        else:
            # Manual progress display
            if current_time - self.last_update_time >= self.update_interval:
                self._manual_progress_display(postfix)
                self.last_update_time = current_time
    
    def _manual_progress_display(self, postfix: Optional[Dict[str, Any]] = None):
        """Manual progress display when tqdm not available"""
        elapsed = time.time() - self.start_time
        
        if self.total > 0:
            percentage = (self.current / self.total) * 100
            bar_length = 40
            filled_length = int(bar_length * self.current // self.total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Calculate rate and ETA
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else float('inf')
            
            # Build display string
            display_parts = [f"{self.description}: |{bar}| {percentage:.1f}%"]
            display_parts.append(f"{self.current}/{self.total}")
            
            if self.show_rate:
                display_parts.append(f"[{rate:.1f} samples/s]")
            
            if self.show_eta and eta != float('inf'):
                eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta/60:.1f}m"
                display_parts.append(f"ETA: {eta_str}")
            
            if postfix:
                postfix_str = ", ".join(f"{k}: {v}" for k, v in postfix.items())
                display_parts.append(f"({postfix_str})")
            
            print(f"\r{' '.join(display_parts)}", end="", flush=True)
    
    def close(self):
        """Close progress bar"""
        if TQDM_AVAILABLE and self.pbar:
            self.pbar.close()
        else:
            print()  # New line after manual progress


@contextmanager
def monitor_operation(operation_name: str, logger: Optional[StructuredLogger] = None,
                     track_resources: bool = True):
    """Context manager for monitoring operations"""
    start_time = time.time()
    start_metrics = None
    
    if track_resources and PSUTIL_AVAILABLE and psutil:
        try:
            process = psutil.Process()
            start_memory = process.memory_info().rss
            start_cpu_time = process.cpu_times()
        except:
            start_memory = 0
            start_cpu_time = None
    else:
        start_memory = 0
        start_cpu_time = None
    
    try:
        if logger:
            logger.info(f"Starting operation: {operation_name}", operation=operation_name)
        
        yield
        
        # Operation completed successfully
        end_time = time.time()
        duration = end_time - start_time
        
        metrics = {"duration_seconds": duration, "status": "success"}
        
        if track_resources and PSUTIL_AVAILABLE and psutil:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss
                memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
                
                metrics.update({
                    "memory_delta_mb": memory_delta,
                    "peak_memory_mb": end_memory / 1024 / 1024
                })
            except:
                pass
        
        if logger:
            logger.info(f"Completed operation: {operation_name}", 
                       operation=operation_name, **metrics)
    
    except Exception as e:
        # Operation failed
        end_time = time.time()
        duration = end_time - start_time
        
        error_metrics = {
            "duration_seconds": duration,
            "status": "failed",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        
        if logger:
            logger.error(f"Failed operation: {operation_name}", 
                        operation=operation_name, **error_metrics, exc_info=True)
        
        raise


def performance_monitor(func: Callable) -> Callable:
    """Decorator for monitoring function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get logger from args/kwargs if available
        logger = None
        for arg in args:
            if isinstance(arg, StructuredLogger):
                logger = arg
                break
        
        if not logger and 'logger' in kwargs:
            logger = kwargs['logger']
        
        function_name = f"{func.__module__}.{func.__name__}"
        
        with monitor_operation(f"Function: {function_name}", logger):
            return func(*args, **kwargs)
    
    return wrapper


class RealTimeMonitor:
    """Real-time monitoring dashboard for synthesis operations"""
    
    def __init__(self, logger: StructuredLogger, update_interval: float = 1.0):
        """Initialize real-time monitor"""
        self.logger = logger
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Data storage
        self.metrics_buffer: List[PerformanceMetrics] = []
        self.max_buffer_size = 1000
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Started real-time monitoring", 
                        update_interval=self.update_interval)
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped real-time monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_current_metrics()
                self.metrics_buffer.append(metrics)
                
                # Trim buffer if too large
                if len(self.metrics_buffer) > self.max_buffer_size:
                    self.metrics_buffer = self.metrics_buffer[-self.max_buffer_size:]
                
                # Log metrics
                self.logger.debug("Real-time metrics update", 
                                metrics=asdict(metrics))
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.update_interval)
    
    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        if PSUTIL_AVAILABLE and psutil:
            try:
                process = psutil.Process()
                
                return PerformanceMetrics(
                    cpu_percent=process.cpu_percent(),
                    memory_mb=process.memory_info().rss / 1024 / 1024,
                    memory_percent=process.memory_percent(),
                    wall_time=time.time()
                )
            except:
                return PerformanceMetrics()
        else:
            return PerformanceMetrics()
    
    def get_live_dashboard_data(self) -> Dict[str, Any]:
        """Get data for live dashboard display"""
        if not self.metrics_buffer:
            return {"status": "No data available"}
        
        recent_metrics = self.metrics_buffer[-60:]  # Last 60 data points
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_metrics": asdict(self.metrics_buffer[-1]),
            "trends": {
                "cpu_trend": [m.cpu_percent for m in recent_metrics],
                "memory_trend": [m.memory_mb for m in recent_metrics],
                "timestamps": [m.timestamp for m in recent_metrics]
            },
            "summary": {
                "avg_cpu": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                "avg_memory": sum(m.memory_mb for m in recent_metrics) / len(recent_metrics),
                "peak_memory": max(m.memory_mb for m in recent_metrics),
                "monitoring_duration": len(self.metrics_buffer) * self.update_interval
            }
        }


# Global logger instance
_global_logger: Optional[StructuredLogger] = None

def get_logger(name: str = "cspbbr3_digital_twin") -> StructuredLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name)
    return _global_logger


if __name__ == "__main__":
    # Demonstration of monitoring capabilities
    logger = get_logger()
    progress_tracker = ProgressTracker(logger)
    
    print("🔍 Demonstrating monitoring capabilities...")
    
    # Test structured logging
    logger.info("System startup", component="monitoring_demo")
    
    # Test progress tracking
    operation_id = progress_tracker.start_operation("demo_op", "Demo Operation", 100)
    
    # Simulate work with progress updates
    with monitor_operation("Demo computation", logger):
        for i in range(100):
            time.sleep(0.01)  # Simulate work
            progress_tracker.update_progress(operation_id, i + 1)
            
            if i % 20 == 0:
                logger.debug(f"Processing step {i}", step=i, progress=i/100)
    
    progress_tracker.complete_operation(operation_id)
    
    # Test performance monitoring
    @performance_monitor
    def sample_computation(n: int, logger: StructuredLogger):
        """Sample computation for testing"""
        import random
        return sum(random.random() for _ in range(n))
    
    result = sample_computation(10000, logger)
    logger.info(f"Computation result: {result}", result=result)
    
    # Display metrics summary
    metrics_summary = logger.get_metrics_summary()
    progress_summary = progress_tracker.get_progress_summary()
    
    print("\n📊 Metrics Summary:")
    print(json.dumps(metrics_summary, indent=2))
    
    print("\n📈 Progress Summary:")
    print(json.dumps(progress_summary, indent=2))
    
    print("\n✅ Monitoring demonstration complete!")
    print(f"📁 Log files created in: {logger.log_dir}")
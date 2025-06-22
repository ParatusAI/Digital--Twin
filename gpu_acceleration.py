#!/usr/bin/env python3
"""
GPU Acceleration Support for CsPbBr3 Digital Twin
Implements CUDA/OpenCL acceleration for computational kernels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import warnings
import time
from pathlib import Path
import platform

# GPU acceleration imports with fallbacks
try:
    import cupy as cp
    import cupyx
    from cupyx.scipy import ndimage as cp_ndimage
    CUPY_AVAILABLE = True
    print("✅ CuPy detected - NVIDIA GPU acceleration available")
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. NVIDIA GPU acceleration disabled.")

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
    print("✅ PyOpenCL detected - OpenCL acceleration available")
except ImportError:
    OPENCL_AVAILABLE = False
    warnings.warn("PyOpenCL not available. OpenCL acceleration disabled.")

try:
    import numba
    from numba import cuda, jit, prange
    NUMBA_AVAILABLE = True
    print("✅ Numba detected - JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. JIT acceleration disabled.")

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA detected - {torch.cuda.device_count()} GPU(s) available")
    else:
        print("⚠️ PyTorch available but no CUDA support")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    warnings.warn("PyTorch not available. Deep learning acceleration disabled.")


class GPUCapabilities:
    """Detect and manage GPU capabilities"""
    
    def __init__(self):
        """Initialize GPU capabilities detection"""
        self.capabilities = self._detect_capabilities()
        
    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect available GPU acceleration capabilities"""
        caps = {
            'cupy_available': CUPY_AVAILABLE,
            'opencl_available': OPENCL_AVAILABLE,
            'numba_cuda_available': NUMBA_AVAILABLE,
            'torch_cuda_available': TORCH_AVAILABLE and torch and torch.cuda.is_available(),
            'cpu_cores': self._get_cpu_cores(),
            'gpu_devices': []
        }
        
        # CuPy GPU detection
        if CUPY_AVAILABLE:
            try:
                gpu_count = cp.cuda.runtime.getDeviceCount()
                for i in range(gpu_count):
                    cp.cuda.Device(i).use()
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    caps['gpu_devices'].append({
                        'id': i,
                        'name': props['name'].decode(),
                        'memory_gb': props['totalGlobalMem'] / 1e9,
                        'compute_capability': f"{props['major']}.{props['minor']}",
                        'backend': 'CUDA'
                    })
            except Exception as e:
                warnings.warn(f"Error detecting CUDA devices: {e}")
        
        # OpenCL device detection
        if OPENCL_AVAILABLE:
            try:
                platforms = cl.get_platforms()
                for platform in platforms:
                    devices = platform.get_devices()
                    for device in devices:
                        caps['gpu_devices'].append({
                            'name': device.name,
                            'memory_gb': device.global_mem_size / 1e9,
                            'compute_units': device.max_compute_units,
                            'backend': 'OpenCL',
                            'platform': platform.name
                        })
            except Exception as e:
                warnings.warn(f"Error detecting OpenCL devices: {e}")
        
        # PyTorch GPU detection
        if caps['torch_cuda_available']:
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    # Check if already detected by CuPy
                    existing = any(d['id'] == i and d['backend'] == 'CUDA' for d in caps['gpu_devices'])
                    if not existing:
                        caps['gpu_devices'].append({
                            'id': i,
                            'name': props.name,
                            'memory_gb': props.total_memory / 1e9,
                            'compute_capability': f"{props.major}.{props.minor}",
                            'backend': 'PyTorch'
                        })
            except Exception as e:
                warnings.warn(f"Error detecting PyTorch CUDA devices: {e}")
        
        return caps
    
    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores"""
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except:
            return 4  # Default fallback
    
    def print_capabilities(self):
        """Print detected GPU capabilities"""
        print("🔍 GPU Acceleration Capabilities:")
        print(f"   CPU Cores: {self.capabilities['cpu_cores']}")
        print(f"   CuPy (NVIDIA): {'✅' if self.capabilities['cupy_available'] else '❌'}")
        print(f"   OpenCL: {'✅' if self.capabilities['opencl_available'] else '❌'}")
        print(f"   Numba CUDA: {'✅' if self.capabilities['numba_cuda_available'] else '❌'}")
        print(f"   PyTorch CUDA: {'✅' if self.capabilities['torch_cuda_available'] else '❌'}")
        
        if self.capabilities['gpu_devices']:
            print("   Available GPU Devices:")
            for i, device in enumerate(self.capabilities['gpu_devices']):
                print(f"     [{i}] {device['name']} ({device['backend']})")
                print(f"         Memory: {device['memory_gb']:.1f} GB")
                if 'compute_capability' in device:
                    print(f"         Compute: {device['compute_capability']}")
        else:
            print("   No GPU devices detected")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get GPU capabilities summary"""
        return {
            'total_devices': len(self.capabilities['gpu_devices']),
            'cupy_available': self.capabilities['cupy_available'],
            'opencl_available': self.capabilities['opencl_available'],
            'numba_cuda_available': self.capabilities['numba_cuda_available'],
            'torch_cuda_available': self.capabilities['torch_cuda_available'],
            'cpu_cores': self.capabilities['cpu_cores'],
            'devices': self.capabilities['gpu_devices']
        }


class GPUArray:
    """Unified GPU array interface supporting multiple backends"""
    
    def __init__(self, data: Union[np.ndarray, 'cp.ndarray', Any], 
                 backend: str = 'auto'):
        """
        Initialize GPU array
        
        Args:
            data: Input data (numpy array, cupy array, or torch tensor)
            backend: GPU backend ('cupy', 'torch', 'auto')
        """
        self.backend = self._determine_backend(data, backend)
        self.data = self._convert_to_backend(data, self.backend)
        
    def _determine_backend(self, data: Any, backend: str) -> str:
        """Determine appropriate backend"""
        if backend != 'auto':
            return backend
        
        # Auto-detect based on data type
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return 'cupy'
        elif TORCH_AVAILABLE and torch and isinstance(data, torch.Tensor):
            return 'torch'
        elif CUPY_AVAILABLE:
            return 'cupy'
        elif TORCH_AVAILABLE and torch and torch.cuda.is_available():
            return 'torch'
        else:
            return 'numpy'
    
    def _convert_to_backend(self, data: Any, backend: str) -> Any:
        """Convert data to specified backend"""
        if isinstance(data, np.ndarray):
            if backend == 'cupy' and CUPY_AVAILABLE:
                return cp.asarray(data)
            elif backend == 'torch' and TORCH_AVAILABLE and torch:
                return torch.from_numpy(data).cuda() if torch.cuda.is_available() else torch.from_numpy(data)
            else:
                return data
        elif CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            if backend == 'torch' and TORCH_AVAILABLE and torch:
                return torch.from_numpy(data.get()).cuda() if torch.cuda.is_available() else torch.from_numpy(data.get())
            elif backend == 'numpy':
                return data.get()
            else:
                return data
        elif TORCH_AVAILABLE and torch and isinstance(data, torch.Tensor):
            if backend == 'cupy' and CUPY_AVAILABLE:
                return cp.asarray(data.detach().cpu().numpy())
            elif backend == 'numpy':
                return data.detach().cpu().numpy()
            else:
                return data
        else:
            return data
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        if self.backend == 'cupy' and CUPY_AVAILABLE:
            return self.data.get()
        elif self.backend == 'torch' and TORCH_AVAILABLE:
            return self.data.detach().cpu().numpy()
        else:
            return self.data
    
    def __getattr__(self, name: str):
        """Delegate attribute access to underlying data"""
        return getattr(self.data, name)


class GPUKernels:
    """Collection of GPU-accelerated computational kernels"""
    
    def __init__(self, backend: str = 'auto'):
        """Initialize GPU kernels"""
        self.backend = backend
        self.caps = GPUCapabilities()
        
        # Initialize backend-specific contexts
        if backend == 'opencl' and OPENCL_AVAILABLE:
            self._init_opencl()
        
    def _init_opencl(self):
        """Initialize OpenCL context"""
        try:
            self.cl_context = cl.create_some_context()
            self.cl_queue = cl.CommandQueue(self.cl_context)
            self._compile_opencl_kernels()
        except Exception as e:
            warnings.warn(f"Failed to initialize OpenCL: {e}")
            self.cl_context = None
            self.cl_queue = None
    
    def _compile_opencl_kernels(self):
        """Compile OpenCL kernels"""
        # Phase probability calculation kernel
        phase_kernel_source = """
        __kernel void calculate_phase_probabilities(
            __global const float* cs_pb_ratio,
            __global const float* temp_normalized,
            __global const float* solvent_effect,
            __global float* prob_3d,
            __global float* prob_0d,
            __global float* prob_2d,
            const int n_samples)
        {
            int idx = get_global_id(0);
            if (idx >= n_samples) return;
            
            float cs_pb = cs_pb_ratio[idx];
            float temp_norm = temp_normalized[idx];
            float solvent_eff = solvent_effect[idx];
            
            // 3D probability
            prob_3d[idx] = 0.6f * (1.0f - fabs(cs_pb - 1.0f)) * 
                          (0.5f + 0.5f * temp_norm) * solvent_eff;
            
            // 0D probability  
            prob_0d[idx] = 0.4f * fmax(0.0f, cs_pb - 1.5f) * 
                          (1.0f - temp_norm) * solvent_eff;
            
            // 2D probability
            prob_2d[idx] = 0.3f * fmax(0.0f, 1.0f / cs_pb - 1.0f) * 
                          temp_norm * solvent_eff;
        }
        """
        
        try:
            self.phase_program = cl.Program(self.cl_context, phase_kernel_source).build()
        except Exception as e:
            warnings.warn(f"Failed to compile OpenCL kernels: {e}")
    
    def vectorized_parameter_generation(self, n_samples: int, 
                                      parameter_bounds: Dict[str, Tuple[float, float]],
                                      backend: str = 'auto') -> Dict[str, np.ndarray]:
        """GPU-accelerated parameter generation"""
        if backend == 'auto':
            backend = 'cupy' if CUPY_AVAILABLE else 'torch' if TORCH_AVAILABLE else 'numpy'
        
        results = {}
        
        if backend == 'cupy' and CUPY_AVAILABLE:
            # CuPy implementation
            for param_name, (min_val, max_val) in parameter_bounds.items():
                if 'concentration' in param_name:
                    # Log-normal distribution
                    log_mean = cp.log((min_val + max_val) / 2)
                    samples = cp.random.lognormal(log_mean, 0.5, size=n_samples)
                    samples = cp.clip(samples, min_val, max_val)
                elif param_name == 'temperature':
                    # Normal distribution
                    samples = cp.random.normal(150.0, 30.0, size=n_samples)
                    samples = cp.clip(samples, min_val, max_val)
                else:
                    # Uniform distribution
                    samples = cp.random.uniform(min_val, max_val, size=n_samples)
                
                results[param_name] = samples.get()  # Convert back to numpy
                
        elif backend == 'torch' and TORCH_AVAILABLE and torch:
            # PyTorch implementation
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            for param_name, (min_val, max_val) in parameter_bounds.items():
                if 'concentration' in param_name:
                    # Log-normal distribution
                    log_mean = np.log((min_val + max_val) / 2)
                    normal_samples = torch.randn(n_samples, device=device)
                    samples = torch.exp(log_mean + 0.5 * normal_samples)
                    samples = torch.clamp(samples, min_val, max_val)
                elif param_name == 'temperature':
                    # Normal distribution
                    samples = torch.normal(150.0, 30.0, size=(n_samples,), device=device)
                    samples = torch.clamp(samples, min_val, max_val)
                else:
                    # Uniform distribution
                    samples = torch.rand(n_samples, device=device) * (max_val - min_val) + min_val
                
                results[param_name] = samples.cpu().numpy()
        
        else:
            # NumPy fallback with optimizations
            if NUMBA_AVAILABLE:
                results = self._numba_parameter_generation(n_samples, parameter_bounds)
            else:
                # Standard numpy implementation
                for param_name, (min_val, max_val) in parameter_bounds.items():
                    if 'concentration' in param_name:
                        log_mean = np.log((min_val + max_val) / 2)
                        samples = np.random.lognormal(log_mean, 0.5, size=n_samples)
                        results[param_name] = np.clip(samples, min_val, max_val)
                    elif param_name == 'temperature':
                        samples = np.random.normal(150.0, 30.0, size=n_samples)
                        results[param_name] = np.clip(samples, min_val, max_val)
                    else:
                        results[param_name] = np.random.uniform(min_val, max_val, size=n_samples)
        
        return results
    
    def _numba_parameter_generation(self, n_samples: int, 
                                  parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """Numba-accelerated parameter generation"""
        
        @numba.jit(nopython=True, parallel=True)
        def generate_lognormal(n, log_mean, sigma, min_val, max_val):
            result = np.empty(n)
            for i in prange(n):
                sample = np.exp(np.random.normal(log_mean, sigma))
                result[i] = min(max(sample, min_val), max_val)
            return result
        
        @numba.jit(nopython=True, parallel=True)
        def generate_normal(n, mean, std, min_val, max_val):
            result = np.empty(n)
            for i in prange(n):
                sample = np.random.normal(mean, std)
                result[i] = min(max(sample, min_val), max_val)
            return result
        
        @numba.jit(nopython=True, parallel=True)
        def generate_uniform(n, min_val, max_val):
            result = np.empty(n)
            for i in prange(n):
                result[i] = np.random.uniform(min_val, max_val)
            return result
        
        results = {}
        for param_name, (min_val, max_val) in parameter_bounds.items():
            if 'concentration' in param_name:
                log_mean = np.log((min_val + max_val) / 2)
                results[param_name] = generate_lognormal(n_samples, log_mean, 0.5, min_val, max_val)
            elif param_name == 'temperature':
                results[param_name] = generate_normal(n_samples, 150.0, 30.0, min_val, max_val)
            else:
                results[param_name] = generate_uniform(n_samples, min_val, max_val)
        
        return results
    
    def accelerated_phase_calculation(self, parameters_df: pd.DataFrame, 
                                    backend: str = 'auto') -> Dict[str, np.ndarray]:
        """GPU-accelerated phase probability calculation"""
        n_samples = len(parameters_df)
        
        # Extract required parameters
        cs_pb_ratio = (parameters_df['cs_br_concentration'] / 
                      (parameters_df['pb_br2_concentration'] + 1e-8)).values
        temp_normalized = ((parameters_df['temperature'] - 80) / (250 - 80)).values
        solvent_effect = np.ones(n_samples) * 1.0  # Simplified
        
        if backend == 'auto':
            backend = 'cupy' if CUPY_AVAILABLE else 'torch' if TORCH_AVAILABLE else 'opencl' if OPENCL_AVAILABLE else 'numpy'
        
        if backend == 'cupy' and CUPY_AVAILABLE:
            # CuPy implementation
            cs_pb_gpu = cp.asarray(cs_pb_ratio)
            temp_norm_gpu = cp.asarray(temp_normalized)
            solvent_eff_gpu = cp.asarray(solvent_effect)
            
            # Vectorized calculations
            prob_3d = 0.6 * (1.0 - cp.abs(cs_pb_gpu - 1.0)) * (0.5 + 0.5 * temp_norm_gpu) * solvent_eff_gpu
            prob_0d = 0.4 * cp.maximum(0, cs_pb_gpu - 1.5) * (1 - temp_norm_gpu) * solvent_eff_gpu
            prob_2d = 0.3 * cp.maximum(0, 1.0 / cs_pb_gpu - 1.0) * temp_norm_gpu * solvent_eff_gpu
            
            # Convert back to numpy
            return {
                'prob_3d': prob_3d.get(),
                'prob_0d': prob_0d.get(),
                'prob_2d': prob_2d.get()
            }
            
        elif backend == 'torch' and TORCH_AVAILABLE and torch:
            # PyTorch implementation
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            cs_pb_tensor = torch.from_numpy(cs_pb_ratio).to(device)
            temp_norm_tensor = torch.from_numpy(temp_normalized).to(device)
            solvent_eff_tensor = torch.from_numpy(solvent_effect).to(device)
            
            prob_3d = 0.6 * (1.0 - torch.abs(cs_pb_tensor - 1.0)) * (0.5 + 0.5 * temp_norm_tensor) * solvent_eff_tensor
            prob_0d = 0.4 * torch.clamp(cs_pb_tensor - 1.5, min=0) * (1 - temp_norm_tensor) * solvent_eff_tensor
            prob_2d = 0.3 * torch.clamp(1.0 / cs_pb_tensor - 1.0, min=0) * temp_norm_tensor * solvent_eff_tensor
            
            return {
                'prob_3d': prob_3d.cpu().numpy(),
                'prob_0d': prob_0d.cpu().numpy(),
                'prob_2d': prob_2d.cpu().numpy()
            }
            
        elif backend == 'opencl' and OPENCL_AVAILABLE and hasattr(self, 'phase_program'):
            # OpenCL implementation
            try:
                # Create buffers
                cs_pb_buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cs_pb_ratio.astype(np.float32))
                temp_norm_buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=temp_normalized.astype(np.float32))
                solvent_eff_buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=solvent_effect.astype(np.float32))
                
                # Output buffers
                prob_3d_buf = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY, cs_pb_ratio.nbytes)
                prob_0d_buf = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY, cs_pb_ratio.nbytes)
                prob_2d_buf = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY, cs_pb_ratio.nbytes)
                
                # Execute kernel
                self.phase_program.calculate_phase_probabilities(
                    self.cl_queue, (n_samples,), None,
                    cs_pb_buf, temp_norm_buf, solvent_eff_buf,
                    prob_3d_buf, prob_0d_buf, prob_2d_buf,
                    np.int32(n_samples)
                )
                
                # Read results
                prob_3d = np.empty_like(cs_pb_ratio, dtype=np.float32)
                prob_0d = np.empty_like(cs_pb_ratio, dtype=np.float32)
                prob_2d = np.empty_like(cs_pb_ratio, dtype=np.float32)
                
                cl.enqueue_copy(self.cl_queue, prob_3d, prob_3d_buf)
                cl.enqueue_copy(self.cl_queue, prob_0d, prob_0d_buf)
                cl.enqueue_copy(self.cl_queue, prob_2d, prob_2d_buf)
                
                return {
                    'prob_3d': prob_3d.astype(np.float64),
                    'prob_0d': prob_0d.astype(np.float64),
                    'prob_2d': prob_2d.astype(np.float64)
                }
                
            except Exception as e:
                warnings.warn(f"OpenCL execution failed: {e}, falling back to NumPy")
                backend = 'numpy'
        
        if backend == 'numpy' or True:  # Fallback
            # NumPy implementation with Numba acceleration
            if NUMBA_AVAILABLE:
                return self._numba_phase_calculation(cs_pb_ratio, temp_normalized, solvent_effect)
            else:
                prob_3d = 0.6 * (1.0 - np.abs(cs_pb_ratio - 1.0)) * (0.5 + 0.5 * temp_normalized) * solvent_effect
                prob_0d = 0.4 * np.maximum(0, cs_pb_ratio - 1.5) * (1 - temp_normalized) * solvent_effect
                prob_2d = 0.3 * np.maximum(0, 1.0 / cs_pb_ratio - 1.0) * temp_normalized * solvent_effect
                
                return {
                    'prob_3d': prob_3d,
                    'prob_0d': prob_0d,
                    'prob_2d': prob_2d
                }
    
    def _numba_phase_calculation(self, cs_pb_ratio: np.ndarray, 
                               temp_normalized: np.ndarray,
                               solvent_effect: np.ndarray) -> Dict[str, np.ndarray]:
        """Numba-accelerated phase calculation"""
        
        @numba.jit(nopython=True, parallel=True)
        def calculate_probabilities(cs_pb, temp_norm, solvent_eff):
            n = len(cs_pb)
            prob_3d = np.empty(n)
            prob_0d = np.empty(n)
            prob_2d = np.empty(n)
            
            for i in prange(n):
                # 3D probability
                prob_3d[i] = 0.6 * (1.0 - abs(cs_pb[i] - 1.0)) * (0.5 + 0.5 * temp_norm[i]) * solvent_eff[i]
                
                # 0D probability
                prob_0d[i] = 0.4 * max(0.0, cs_pb[i] - 1.5) * (1.0 - temp_norm[i]) * solvent_eff[i]
                
                # 2D probability
                prob_2d[i] = 0.3 * max(0.0, 1.0 / cs_pb[i] - 1.0) * temp_norm[i] * solvent_eff[i]
            
            return prob_3d, prob_0d, prob_2d
        
        prob_3d, prob_0d, prob_2d = calculate_probabilities(cs_pb_ratio, temp_normalized, solvent_effect)
        
        return {
            'prob_3d': prob_3d,
            'prob_0d': prob_0d,
            'prob_2d': prob_2d
        }


class GPUDataGenerator:
    """GPU-accelerated data generation pipeline"""
    
    def __init__(self, backend: str = 'auto'):
        """Initialize GPU data generator"""
        self.backend = backend
        self.kernels = GPUKernels(backend)
        self.caps = GPUCapabilities()
        
    def generate_large_dataset(self, n_samples: int, 
                             parameter_bounds: Dict[str, Tuple[float, float]],
                             batch_size: Optional[int] = None) -> pd.DataFrame:
        """Generate large dataset using GPU acceleration"""
        if batch_size is None:
            # Estimate optimal batch size based on available memory
            if self.caps.capabilities['gpu_devices']:
                # Assume 4 bytes per float, 10 parameters, safety factor of 0.1
                available_memory = min(d['memory_gb'] for d in self.caps.capabilities['gpu_devices']) * 1e9
                batch_size = int(available_memory * 0.1 / (10 * 4))
                batch_size = min(batch_size, n_samples, 1000000)  # Cap at 1M
            else:
                batch_size = min(100000, n_samples)  # CPU fallback
        
        print(f"🚀 Generating {n_samples} samples using {self.backend} backend")
        print(f"   Batch size: {batch_size}")
        
        all_data = []
        start_time = time.time()
        
        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            
            # Generate parameters
            batch_params = self.kernels.vectorized_parameter_generation(
                current_batch_size, parameter_bounds, self.backend
            )
            
            # Create DataFrame for this batch
            batch_df = pd.DataFrame(batch_params)
            
            # Calculate physics features
            batch_physics = self._calculate_batch_physics(batch_df)
            batch_df = pd.concat([batch_df, pd.DataFrame(batch_physics)], axis=1)
            
            # Calculate phase probabilities
            phase_probs = self.kernels.accelerated_phase_calculation(batch_df, self.backend)
            batch_df = pd.concat([batch_df, pd.DataFrame(phase_probs)], axis=1)
            
            all_data.append(batch_df)
            
            if (i // batch_size) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + current_batch_size) / elapsed if elapsed > 0 else 0
                print(f"   Progress: {i + current_batch_size}/{n_samples} ({rate:.0f} samples/sec)")
        
        # Combine all batches
        final_df = pd.concat(all_data, ignore_index=True)
        
        elapsed = time.time() - start_time
        rate = n_samples / elapsed if elapsed > 0 else 0
        print(f"✅ Generated {len(final_df)} samples in {elapsed:.1f}s ({rate:.0f} samples/sec)")
        
        return final_df
    
    def _calculate_batch_physics(self, batch_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate physics features for batch"""
        n_samples = len(batch_df)
        
        # Simple physics calculations (could be further GPU-accelerated)
        if CUPY_AVAILABLE and self.backend == 'cupy':
            # CuPy implementation
            cs_conc = cp.asarray(batch_df['cs_br_concentration'].values)
            pb_conc = cp.asarray(batch_df['pb_br2_concentration'].values)
            temp = cp.asarray(batch_df['temperature'].values)
            
            # Physics calculations
            temp_kelvin = temp + 273.15
            supersaturation = cs_conc * pb_conc * cp.exp(-5000 / (8.314 * temp_kelvin))
            nucleation_rate = 1e10 * supersaturation**2 * cp.exp(-temp / 100)
            growth_rate = supersaturation * temp / (1.0 + 0.1 * 10.0)  # Simplified
            
            return {
                'supersaturation': supersaturation.get(),
                'nucleation_rate': nucleation_rate.get(),
                'growth_rate': growth_rate.get(),
                'solvent_effect': cp.ones(n_samples).get()
            }
            
        else:
            # NumPy fallback
            cs_conc = batch_df['cs_br_concentration'].values
            pb_conc = batch_df['pb_br2_concentration'].values
            temp = batch_df['temperature'].values
            
            temp_kelvin = temp + 273.15
            supersaturation = cs_conc * pb_conc * np.exp(-5000 / (8.314 * temp_kelvin))
            nucleation_rate = 1e10 * supersaturation**2 * np.exp(-temp / 100)
            growth_rate = supersaturation * temp / (1.0 + 0.1 * 10.0)
            
            return {
                'supersaturation': supersaturation,
                'nucleation_rate': nucleation_rate,
                'growth_rate': growth_rate,
                'solvent_effect': np.ones(n_samples)
            }
    
    def benchmark_performance(self, sizes: List[int] = [1000, 10000, 100000]) -> Dict[str, Any]:
        """Benchmark GPU vs CPU performance"""
        results = {
            'sizes': sizes,
            'backends': {},
            'speedups': {}
        }
        
        parameter_bounds = {
            'cs_br_concentration': (0.1, 3.0),
            'pb_br2_concentration': (0.1, 2.0),
            'temperature': (80.0, 250.0)
        }
        
        # Test available backends
        backends_to_test = ['numpy']
        if CUPY_AVAILABLE:
            backends_to_test.append('cupy')
        if TORCH_AVAILABLE and torch and torch.cuda.is_available():
            backends_to_test.append('torch')
        if NUMBA_AVAILABLE:
            backends_to_test.append('numba')
        
        print("🏁 Benchmarking GPU performance...")
        
        for backend in backends_to_test:
            results['backends'][backend] = []
            
            for size in sizes:
                print(f"   Testing {backend} with {size} samples...")
                
                start_time = time.time()
                
                if backend == 'numba':
                    # Use NumPy with Numba acceleration
                    params = self.kernels._numba_parameter_generation(size, parameter_bounds)
                else:
                    params = self.kernels.vectorized_parameter_generation(size, parameter_bounds, backend)
                
                elapsed = time.time() - start_time
                rate = size / elapsed if elapsed > 0 else 0
                
                results['backends'][backend].append({
                    'size': size,
                    'time': elapsed,
                    'rate': rate
                })
                
                print(f"     {elapsed:.3f}s ({rate:.0f} samples/sec)")
        
        # Calculate speedups relative to NumPy
        if 'numpy' in results['backends']:
            numpy_times = {r['size']: r['time'] for r in results['backends']['numpy']}
            
            for backend, backend_results in results['backends'].items():
                if backend != 'numpy':
                    speedups = []
                    for result in backend_results:
                        size = result['size']
                        if size in numpy_times and numpy_times[size] > 0:
                            speedup = numpy_times[size] / result['time']
                            speedups.append(speedup)
                        else:
                            speedups.append(1.0)
                    results['speedups'][backend] = speedups
        
        return results


def create_performance_report(benchmark_results: Dict[str, Any], 
                            output_path: str = "gpu_performance_report.html"):
    """Create GPU performance analysis report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPU Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .benchmark-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .speedup-good {{ color: #27ae60; font-weight: bold; }}
            .speedup-moderate {{ color: #f39c12; }}
            .speedup-poor {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <h1>🚀 GPU Performance Analysis Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="benchmark-section">
            <h2>📊 Performance Benchmarks</h2>
            <table>
                <tr><th>Backend</th><th>Dataset Size</th><th>Time (s)</th><th>Rate (samples/s)</th><th>Speedup vs NumPy</th></tr>
    """
    
    # Add benchmark results
    for backend, results in benchmark_results['backends'].items():
        for i, result in enumerate(results):
            speedup = ""
            if backend in benchmark_results['speedups'] and i < len(benchmark_results['speedups'][backend]):
                speedup_val = benchmark_results['speedups'][backend][i]
                speedup_class = 'speedup-good' if speedup_val > 2 else 'speedup-moderate' if speedup_val > 1.2 else 'speedup-poor'
                speedup = f"<span class='{speedup_class}'>{speedup_val:.1f}x</span>"
            
            html_content += f"""
                <tr>
                    <td>{backend}</td>
                    <td>{result['size']:,}</td>
                    <td>{result['time']:.3f}</td>
                    <td>{result['rate']:,.0f}</td>
                    <td>{speedup}</td>
                </tr>
            """
    
    html_content += """
            </table>
        </div>
        
        <div class="benchmark-section">
            <h2>💡 Recommendations</h2>
            <ul>
                <li>Use GPU acceleration for datasets larger than 10,000 samples</li>
                <li>CuPy generally provides the best performance for NVIDIA GPUs</li>
                <li>PyTorch is good for deep learning integration</li>
                <li>Numba provides significant speedups even without GPU</li>
                <li>Consider batch processing for very large datasets</li>
            </ul>
        </div>
        
        <footer style="margin-top: 40px; text-align: center; color: #6c757d;">
            <p>Generated by CsPbBr₃ Digital Twin GPU Acceleration System</p>
        </footer>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"📊 Performance report saved to: {output_path}")


if __name__ == "__main__":
    # Demonstration of GPU acceleration capabilities
    print("🚀 GPU Acceleration Demo")
    
    # Detect capabilities
    caps = GPUCapabilities()
    caps.print_capabilities()
    
    # Initialize GPU kernels
    kernels = GPUKernels()
    
    # Test parameter generation
    print("\n🧪 Testing parameter generation...")
    parameter_bounds = {
        'cs_br_concentration': (0.1, 3.0),
        'pb_br2_concentration': (0.1, 2.0),
        'temperature': (80.0, 250.0),
        'oa_concentration': (0.0, 1.5),
        'oam_concentration': (0.0, 1.5),
        'reaction_time': (1.0, 60.0)
    }
    
    # Generate test dataset
    test_size = 10000
    start_time = time.time()
    
    params = kernels.vectorized_parameter_generation(test_size, parameter_bounds)
    param_df = pd.DataFrame(params)
    
    # Test phase calculation
    phase_results = kernels.accelerated_phase_calculation(param_df)
    
    elapsed = time.time() - start_time
    rate = test_size / elapsed if elapsed > 0 else 0
    
    print(f"   Generated {test_size} samples in {elapsed:.3f}s ({rate:.0f} samples/sec)")
    print(f"   Parameters shape: {param_df.shape}")
    print(f"   Phase probabilities calculated for {len(phase_results)} properties")
    
    # Benchmark performance
    print("\n🏁 Running performance benchmark...")
    data_generator = GPUDataGenerator()
    benchmark_results = data_generator.benchmark_performance([1000, 10000, 50000])
    
    # Create performance report
    create_performance_report(benchmark_results)
    
    # Test large dataset generation
    if caps.capabilities['gpu_devices'] or NUMBA_AVAILABLE:
        print("\n📊 Testing large dataset generation...")
        large_dataset = data_generator.generate_large_dataset(
            100000, parameter_bounds, batch_size=25000
        )
        print(f"   Final dataset shape: {large_dataset.shape}")
        
        # Save sample
        large_dataset.head(1000).to_csv("gpu_generated_sample.csv", index=False)
        print("   Sample saved to: gpu_generated_sample.csv")
    
    print("\n✅ GPU acceleration demo complete!")
    print("📁 Generated files:")
    print("   - gpu_performance_report.html")
    if caps.capabilities['gpu_devices'] or NUMBA_AVAILABLE:
        print("   - gpu_generated_sample.csv")
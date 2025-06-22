#!/usr/bin/env python3
"""
Comprehensive Diagnostics Script for CsPbBr3 Digital Twin
Checks for syntax errors, import issues, dependencies, and system problems
"""

import os
import sys
import ast
import subprocess
import importlib
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

class ComprehensiveDiagnostics:
    """Comprehensive diagnostics for the entire codebase"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            'syntax_errors': [],
            'import_errors': [],
            'dependency_issues': [],
            'file_issues': [],
            'system_info': {},
            'recommendations': []
        }
        
    def run_all_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic checks"""
        print("🔍 Running Comprehensive Diagnostics...")
        print("=" * 60)
        
        # 1. System and environment checks
        print("1️⃣ Checking System Environment...")
        self.check_system_environment()
        
        # 2. Python environment checks
        print("\n2️⃣ Checking Python Environment...")
        self.check_python_environment()
        
        # 3. Dependency checks
        print("\n3️⃣ Checking Dependencies...")
        self.check_dependencies()
        
        # 4. File structure checks
        print("\n4️⃣ Checking File Structure...")
        self.check_file_structure()
        
        # 5. Syntax checks
        print("\n5️⃣ Checking Syntax Errors...")
        self.check_syntax_errors()
        
        # 6. Import checks
        print("\n6️⃣ Checking Import Issues...")
        self.check_import_issues()
        
        # 7. Runtime checks
        print("\n7️⃣ Checking Runtime Issues...")
        self.check_runtime_issues()
        
        # 8. Configuration checks
        print("\n8️⃣ Checking Configuration...")
        self.check_configuration()
        
        # 9. Generate recommendations
        print("\n9️⃣ Generating Recommendations...")
        self.generate_recommendations()
        
        # 10. Generate report
        print("\n🔟 Generating Diagnostic Report...")
        self.generate_report()
        
        return self.results
    
    def check_system_environment(self):
        """Check system environment and resources"""
        try:
            # Python version
            python_version = sys.version
            self.results['system_info']['python_version'] = python_version
            print(f"   Python Version: {python_version.split()[0]}")
            
            # Platform
            import platform
            self.results['system_info']['platform'] = platform.platform()
            print(f"   Platform: {platform.platform()}")
            
            # Memory usage (if psutil available)
            try:
                import psutil
                memory = psutil.virtual_memory()
                self.results['system_info']['memory_total_gb'] = memory.total / (1024**3)
                self.results['system_info']['memory_available_gb'] = memory.available / (1024**3)
                print(f"   Memory: {memory.available/(1024**3):.1f}GB available / {memory.total/(1024**3):.1f}GB total")
            except ImportError:
                print("   Memory: psutil not available")
            
            # Disk space
            try:
                disk_usage = os.statvfs(self.project_root)
                available_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
                total_gb = (disk_usage.f_blocks * disk_usage.f_frsize) / (1024**3)
                self.results['system_info']['disk_available_gb'] = available_gb
                self.results['system_info']['disk_total_gb'] = total_gb
                print(f"   Disk: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
            except:
                print("   Disk: Could not check disk space")
            
            # CPU count
            cpu_count = os.cpu_count()
            self.results['system_info']['cpu_count'] = cpu_count
            print(f"   CPU Cores: {cpu_count}")
            
        except Exception as e:
            error = f"System environment check failed: {e}"
            self.results['system_info']['error'] = error
            print(f"   ❌ {error}")
    
    def check_python_environment(self):
        """Check Python environment and paths"""
        try:
            # Python executable
            python_exe = sys.executable
            self.results['system_info']['python_executable'] = python_exe
            print(f"   Python Executable: {python_exe}")
            
            # Python path
            python_path = sys.path
            self.results['system_info']['python_path'] = python_path
            print(f"   Python Path: {len(python_path)} directories")
            
            # Virtual environment check
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            self.results['system_info']['in_virtual_env'] = in_venv
            print(f"   Virtual Environment: {'Yes' if in_venv else 'No'}")
            
            if in_venv:
                print(f"   Virtual Env Path: {sys.prefix}")
            
            # Site packages
            try:
                import site
                site_packages = site.getsitepackages()
                self.results['system_info']['site_packages'] = site_packages
                print(f"   Site Packages: {len(site_packages)} locations")
            except:
                print("   Site Packages: Could not determine")
                
        except Exception as e:
            error = f"Python environment check failed: {e}"
            self.results['system_info']['python_error'] = error
            print(f"   ❌ {error}")
    
    def check_dependencies(self):
        """Check all required dependencies"""
        required_packages = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly',
            'scikit-learn', 'scipy', 'psutil', 'jupyter', 'ipywidgets'
        ]
        
        optional_packages = [
            'cupy', 'pyopencl', 'numba', 'torch', 'git'
        ]
        
        for package in required_packages:
            self._check_package(package, required=True)
        
        for package in optional_packages:
            self._check_package(package, required=False)
    
    def _check_package(self, package_name: str, required: bool = True):
        """Check if a specific package is available"""
        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            
            # Get version if available
            version = getattr(module, '__version__', 'Unknown')
            
            print(f"   {'✅' if required else '🟦'} {package_name}: {version}")
            
            # Store result
            self.results['dependency_issues'].append({
                'package': package_name,
                'status': 'available',
                'version': version,
                'required': required
            })
            
        except ImportError as e:
            status = '❌' if required else '⚠️'
            print(f"   {status} {package_name}: Not available ({e})")
            
            self.results['dependency_issues'].append({
                'package': package_name,
                'status': 'missing',
                'error': str(e),
                'required': required
            })
        
        except Exception as e:
            print(f"   ⚠️ {package_name}: Error checking ({e})")
            
            self.results['dependency_issues'].append({
                'package': package_name,
                'status': 'error',
                'error': str(e),
                'required': required
            })
    
    def check_file_structure(self):
        """Check file structure and permissions"""
        try:
            # Find all Python files
            python_files = list(self.project_root.glob("**/*.py"))
            self.results['system_info']['python_files_count'] = len(python_files)
            print(f"   Found {len(python_files)} Python files")
            
            # Check key files exist
            key_files = [
                'generate_sample_data.py',
                'ml_optimization.py',
                'adaptive_sampling.py',
                'monitoring.py',
                'gpu_acceleration.py',
                'advanced_physics.py',
                'data_versioning.py',
                'experimental_integration.py',
                'interactive_exploration.py',
                'enhanced_training.py',
                'working_enhanced_training.py'
            ]
            
            missing_files = []
            for file_name in key_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    print(f"   ✅ {file_name}")
                else:
                    print(f"   ❌ {file_name} - Missing")
                    missing_files.append(file_name)
            
            self.results['file_issues'] = missing_files
            
            # Check config directory
            config_dir = self.project_root / 'config'
            if config_dir.exists():
                print(f"   ✅ config/ directory exists")
                config_files = list(config_dir.glob("*.json"))
                print(f"   📁 {len(config_files)} config files found")
            else:
                print(f"   ⚠️ config/ directory missing")
                self.results['file_issues'].append('config directory missing')
            
            # Check permissions
            for file_path in python_files[:5]:  # Check first 5 files
                if os.access(file_path, os.R_OK):
                    print(f"   ✅ {file_path.name} readable")
                else:
                    print(f"   ❌ {file_path.name} not readable")
                    self.results['file_issues'].append(f'{file_path.name} not readable')
            
        except Exception as e:
            error = f"File structure check failed: {e}"
            print(f"   ❌ {error}")
            self.results['file_issues'].append(error)
    
    def check_syntax_errors(self):
        """Check all Python files for syntax errors"""
        python_files = list(self.project_root.glob("**/*.py"))
        
        syntax_errors = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Parse the AST to check syntax
                ast.parse(source_code, filename=str(file_path))
                print(f"   ✅ {file_path.name}")
                
            except SyntaxError as e:
                error_info = {
                    'file': str(file_path),
                    'line': e.lineno,
                    'column': e.offset,
                    'message': e.msg,
                    'text': e.text
                }
                syntax_errors.append(error_info)
                print(f"   ❌ {file_path.name}: Syntax error at line {e.lineno}")
                
            except Exception as e:
                error_info = {
                    'file': str(file_path),
                    'error': str(e)
                }
                syntax_errors.append(error_info)
                print(f"   ⚠️ {file_path.name}: Could not parse ({e})")
        
        self.results['syntax_errors'] = syntax_errors
        
        if not syntax_errors:
            print("   🎉 No syntax errors found!")
    
    def check_import_issues(self):
        """Check import issues in key modules"""
        key_modules = [
            'generate_sample_data',
            'ml_optimization',
            'adaptive_sampling',
            'monitoring',
            'gpu_acceleration',
            'advanced_physics',
            'data_versioning',
            'experimental_integration',
            'interactive_exploration',
            'working_enhanced_training'
        ]
        
        import_errors = []
        
        # Add current directory to path for imports
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        for module_name in key_modules:
            try:
                # Try to import the module
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    module = importlib.import_module(module_name)
                
                print(f"   ✅ {module_name}")
                
                # Try to reload to catch any import issues
                importlib.reload(module)
                
            except ImportError as e:
                error_info = {
                    'module': module_name,
                    'error_type': 'ImportError',
                    'message': str(e)
                }
                import_errors.append(error_info)
                print(f"   ❌ {module_name}: ImportError - {e}")
                
            except Exception as e:
                error_info = {
                    'module': module_name,
                    'error_type': type(e).__name__,
                    'message': str(e)
                }
                import_errors.append(error_info)
                print(f"   ⚠️ {module_name}: {type(e).__name__} - {e}")
        
        self.results['import_errors'] = import_errors
        
        if not import_errors:
            print("   🎉 No import errors found!")
    
    def check_runtime_issues(self):
        """Check for runtime issues by running basic functionality"""
        runtime_errors = []
        
        try:
            # Test basic data generation
            print("   Testing basic data generation...")
            from simple_training_test import run_simple_training
            
            # Run a small test
            test_data = run_simple_training(10)  # Just 10 samples
            if test_data and len(test_data) == 10:
                print("   ✅ Basic data generation works")
            else:
                runtime_errors.append("Basic data generation failed")
                print("   ❌ Basic data generation failed")
                
        except Exception as e:
            runtime_errors.append(f"Data generation test failed: {e}")
            print(f"   ❌ Data generation test failed: {e}")
        
        try:
            # Test working enhanced training
            print("   Testing enhanced training pipeline...")
            from working_enhanced_training import WorkingEnhancedTrainer
            
            trainer = WorkingEnhancedTrainer(output_dir="test_output")
            # Test just the initialization and small dataset
            test_samples = trainer._generate_random_sample()
            test_result = trainer.simulate_physics_enhanced(test_samples)
            
            if 'plqy' in test_result and 'bandgap' in test_result:
                print("   ✅ Enhanced training pipeline works")
            else:
                runtime_errors.append("Enhanced training pipeline failed")
                print("   ❌ Enhanced training pipeline failed")
                
        except Exception as e:
            runtime_errors.append(f"Enhanced training test failed: {e}")
            print(f"   ❌ Enhanced training test failed: {e}")
        
        self.results['runtime_errors'] = runtime_errors
        
        if not runtime_errors:
            print("   🎉 No runtime errors found!")
    
    def check_configuration(self):
        """Check configuration files and settings"""
        config_issues = []
        
        try:
            # Check if config directory exists
            config_dir = self.project_root / 'config'
            if not config_dir.exists():
                config_issues.append("Config directory missing")
                print("   ❌ Config directory missing")
            else:
                print("   ✅ Config directory exists")
                
                # Check for config files
                config_file = config_dir / 'data_generation_config.json'
                if config_file.exists():
                    print("   ✅ data_generation_config.json exists")
                    
                    # Try to load and validate config
                    try:
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                        
                        required_keys = ['synthesis_parameters', 'solvents']
                        missing_keys = [key for key in required_keys if key not in config_data]
                        
                        if missing_keys:
                            config_issues.append(f"Config missing keys: {missing_keys}")
                            print(f"   ❌ Config missing keys: {missing_keys}")
                        else:
                            print("   ✅ Config file is valid")
                            
                    except json.JSONDecodeError as e:
                        config_issues.append(f"Config file invalid JSON: {e}")
                        print(f"   ❌ Config file invalid JSON: {e}")
                        
                else:
                    config_issues.append("data_generation_config.json missing")
                    print("   ❌ data_generation_config.json missing")
            
            # Check requirements.txt
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                print("   ✅ requirements.txt exists")
            else:
                config_issues.append("requirements.txt missing")
                print("   ⚠️ requirements.txt missing")
        
        except Exception as e:
            config_issues.append(f"Configuration check failed: {e}")
            print(f"   ❌ Configuration check failed: {e}")
        
        self.results['config_issues'] = config_issues
    
    def generate_recommendations(self):
        """Generate recommendations based on diagnostic results"""
        recommendations = []
        
        # Check for missing required dependencies
        missing_required = [
            dep for dep in self.results['dependency_issues'] 
            if dep['required'] and dep['status'] == 'missing'
        ]
        
        if missing_required:
            missing_names = [dep['package'] for dep in missing_required]
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Dependencies',
                'issue': f'Missing required packages: {", ".join(missing_names)}',
                'solution': f'Run: pip install {" ".join(missing_names)}'
            })
        
        # Check for syntax errors
        if self.results['syntax_errors']:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Syntax',
                'issue': f'{len(self.results["syntax_errors"])} syntax errors found',
                'solution': 'Fix syntax errors in the reported files'
            })
        
        # Check for import errors
        if self.results['import_errors']:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Imports',
                'issue': f'{len(self.results["import_errors"])} import errors found',
                'solution': 'Check dependency installation and file paths'
            })
        
        # Check for file issues
        if self.results['file_issues']:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Files',
                'issue': f'{len(self.results["file_issues"])} file issues found',
                'solution': 'Check file permissions and missing files'
            })
        
        # Performance recommendations
        memory_gb = self.results['system_info'].get('memory_available_gb', 0)
        if memory_gb < 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Performance',
                'issue': f'Low memory: {memory_gb:.1f}GB available',
                'solution': 'Consider reducing dataset size or upgrading memory'
            })
        
        # Virtual environment recommendation
        if not self.results['system_info'].get('in_virtual_env', False):
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Environment',
                'issue': 'Not using virtual environment',
                'solution': 'Consider using virtual environment for better dependency isolation'
            })
        
        # Config recommendations
        if hasattr(self.results, 'config_issues') and self.results.get('config_issues'):
            recommendations.append({
                'priority': 'LOW',
                'category': 'Configuration',
                'issue': 'Configuration issues found',
                'solution': 'Create missing config files and fix JSON syntax'
            })
        
        self.results['recommendations'] = recommendations
        
        # Print recommendations
        if recommendations:
            print("   📋 Recommendations generated:")
            for rec in recommendations:
                priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[rec['priority']]
                print(f"   {priority_emoji} {rec['priority']}: {rec['issue']}")
        else:
            print("   🎉 No recommendations needed!")
    
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        report_path = self.project_root / "diagnostic_report.html"
        
        # Count issues
        total_syntax_errors = len(self.results['syntax_errors'])
        total_import_errors = len(self.results['import_errors'])
        total_file_issues = len(self.results['file_issues'])
        total_recommendations = len(self.results['recommendations'])
        
        # Determine overall status
        if total_syntax_errors > 0 or total_import_errors > 0:
            overall_status = "🔴 CRITICAL ISSUES"
            status_class = "critical"
        elif total_file_issues > 0 or total_recommendations > 0:
            overall_status = "🟡 MINOR ISSUES"
            status_class = "warning"
        else:
            overall_status = "🟢 ALL GOOD"
            status_class = "good"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CsPbBr3 Digital Twin - Diagnostic Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background-color: #e9ecef; border-radius: 5px; min-width: 150px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #6c757d; }}
                .status-critical {{ background-color: #f8d7da; color: #721c24; }}
                .status-warning {{ background-color: #fff3cd; color: #856404; }}
                .status-good {{ background-color: #d4edda; color: #155724; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .error {{ background-color: #f8d7da; }}
                .warning {{ background-color: #fff3cd; }}
                .success {{ background-color: #d4edda; }}
                .code {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <h1>🔍 CsPbBr3 Digital Twin - Comprehensive Diagnostic Report</h1>
            <p><strong>Generated:</strong> {self._get_timestamp()}</p>
            
            <div class="summary status-{status_class}">
                <h2>📊 Overall Status: {overall_status}</h2>
                <div class="metric">
                    <div class="metric-value">{total_syntax_errors}</div>
                    <div class="metric-label">Syntax Errors</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_import_errors}</div>
                    <div class="metric-label">Import Errors</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_file_issues}</div>
                    <div class="metric-label">File Issues</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_recommendations}</div>
                    <div class="metric-label">Recommendations</div>
                </div>
            </div>
        """
        
        # System Information
        html_content += """
            <h2>💻 System Information</h2>
            <table>
        """
        
        for key, value in self.results['system_info'].items():
            if not key.endswith('_error'):
                html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html_content += "</table>"
        
        # Dependencies
        html_content += """
            <h2>📦 Dependencies</h2>
            <table>
                <tr><th>Package</th><th>Status</th><th>Version</th><th>Required</th></tr>
        """
        
        for dep in self.results['dependency_issues']:
            status_class = 'success' if dep['status'] == 'available' else 'error'
            required_text = 'Yes' if dep['required'] else 'No'
            version_text = dep.get('version', dep.get('error', 'N/A'))
            
            html_content += f"""
                <tr class="{status_class}">
                    <td>{dep['package']}</td>
                    <td>{dep['status'].title()}</td>
                    <td>{version_text}</td>
                    <td>{required_text}</td>
                </tr>
            """
        
        html_content += "</table>"
        
        # Syntax Errors
        if self.results['syntax_errors']:
            html_content += """
                <h2>❌ Syntax Errors</h2>
                <table>
                    <tr><th>File</th><th>Line</th><th>Error</th></tr>
            """
            
            for error in self.results['syntax_errors']:
                html_content += f"""
                    <tr class="error">
                        <td>{Path(error['file']).name}</td>
                        <td>{error.get('line', 'N/A')}</td>
                        <td>{error.get('message', error.get('error', 'Unknown error'))}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Import Errors
        if self.results['import_errors']:
            html_content += """
                <h2>📥 Import Errors</h2>
                <table>
                    <tr><th>Module</th><th>Error Type</th><th>Message</th></tr>
            """
            
            for error in self.results['import_errors']:
                html_content += f"""
                    <tr class="error">
                        <td>{error['module']}</td>
                        <td>{error['error_type']}</td>
                        <td>{error['message']}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Recommendations
        if self.results['recommendations']:
            html_content += """
                <h2>💡 Recommendations</h2>
                <table>
                    <tr><th>Priority</th><th>Category</th><th>Issue</th><th>Solution</th></tr>
            """
            
            for rec in self.results['recommendations']:
                priority_class = {
                    'HIGH': 'error',
                    'MEDIUM': 'warning',
                    'LOW': 'success'
                }[rec['priority']]
                
                html_content += f"""
                    <tr class="{priority_class}">
                        <td>{rec['priority']}</td>
                        <td>{rec['category']}</td>
                        <td>{rec['issue']}</td>
                        <td><div class="code">{rec['solution']}</div></td>
                    </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
            <footer style="margin-top: 40px; text-align: center; color: #6c757d;">
                <p>Generated by CsPbBr3 Digital Twin Comprehensive Diagnostics</p>
            </footer>
        </body>
        </html>
        """
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"   📄 Diagnostic report saved to: {report_path}")
        
        # Also save JSON report
        json_path = self.project_root / "diagnostic_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"   📊 JSON results saved to: {json_path}")
        
        return str(report_path)
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def print_summary(self):
        """Print a summary of diagnostic results"""
        print("\n" + "="*60)
        print("🏁 DIAGNOSTIC SUMMARY")
        print("="*60)
        
        # Count issues
        syntax_count = len(self.results['syntax_errors'])
        import_count = len(self.results['import_errors'])
        file_count = len(self.results['file_issues'])
        rec_count = len(self.results['recommendations'])
        
        if syntax_count == 0 and import_count == 0:
            print("🎉 SUCCESS: No critical errors found!")
        else:
            print("⚠️  ISSUES FOUND:")
            
        if syntax_count > 0:
            print(f"   ❌ {syntax_count} syntax errors")
        if import_count > 0:
            print(f"   ❌ {import_count} import errors")
        if file_count > 0:
            print(f"   ⚠️ {file_count} file issues")
        if rec_count > 0:
            print(f"   💡 {rec_count} recommendations")
        
        # Quick fixes
        missing_required = [
            dep['package'] for dep in self.results['dependency_issues'] 
            if dep['required'] and dep['status'] == 'missing'
        ]
        
        if missing_required:
            print(f"\n🔧 QUICK FIX - Install missing packages:")
            print(f"   pip install {' '.join(missing_required)}")
        
        print(f"\n📄 Full report available in diagnostic_report.html")


def main():
    """Main function to run diagnostics"""
    print("🔍 CsPbBr3 Digital Twin - Comprehensive Diagnostics")
    print("=" * 60)
    
    # Run diagnostics
    diagnostics = ComprehensiveDiagnostics()
    results = diagnostics.run_all_diagnostics()
    
    # Print summary
    diagnostics.print_summary()
    
    return results


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Data Versioning and Lineage Tracking for CsPbBr3 Digital Twin
Implements comprehensive data provenance, versioning, and reproducibility tracking
"""

import hashlib
import json
import pickle
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field, asdict
import warnings
import shutil
import os

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Some features will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Some features will be limited.")

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    warnings.warn("GitPython not available. Git integration disabled.")


@dataclass
class DataVersion:
    """Container for data version information"""
    version_id: str
    parent_version: Optional[str]
    created_at: str
    created_by: str
    description: str
    data_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    size_bytes: int = 0


@dataclass
class ProcessingStep:
    """Container for data processing step information"""
    step_id: str
    step_type: str
    function_name: str
    parameters: Dict[str, Any]
    input_versions: List[str]
    output_version: str
    execution_time: float
    timestamp: str
    git_commit: Optional[str] = None
    environment_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageNode:
    """Node in the data lineage graph"""
    node_id: str
    node_type: str  # 'data', 'process', 'parameter'
    name: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class LineageEdge:
    """Edge in the data lineage graph"""
    edge_id: str
    source_node: str
    target_node: str
    edge_type: str  # 'input', 'output', 'parameter', 'derived_from'
    properties: Dict[str, Any] = field(default_factory=dict)


class DataVersionManager:
    """Manages data versions and maintains version history"""
    
    def __init__(self, storage_dir: str = "data_versions"):
        """Initialize data version manager"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.storage_dir / "versions.db"
        self._init_database()
        
        # Initialize metadata storage
        self.metadata_dir = self.storage_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Initialize data storage
        self.data_dir = self.storage_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
    def _init_database(self):
        """Initialize SQLite database for version tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_versions (
                    version_id TEXT PRIMARY KEY,
                    parent_version TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT,
                    data_hash TEXT NOT NULL,
                    file_path TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    metadata TEXT,
                    tags TEXT,
                    FOREIGN KEY (parent_version) REFERENCES data_versions (version_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_steps (
                    step_id TEXT PRIMARY KEY,
                    step_type TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    parameters TEXT,
                    input_versions TEXT,
                    output_version TEXT NOT NULL,
                    execution_time REAL,
                    timestamp TEXT NOT NULL,
                    git_commit TEXT,
                    environment_info TEXT,
                    FOREIGN KEY (output_version) REFERENCES data_versions (version_id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_created_at ON data_versions (created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_hash ON data_versions (data_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_timestamp ON processing_steps (timestamp)")
            
            conn.commit()
    
    def create_version(self, data: Any, description: str, 
                      parent_version: Optional[str] = None,
                      created_by: str = "system",
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new data version"""
        # Generate unique version ID
        version_id = f"v_{uuid.uuid4().hex[:12]}"
        
        # Calculate data hash
        data_hash = self._calculate_hash(data)
        
        # Check for existing version with same hash
        existing_version = self._find_version_by_hash(data_hash)
        if existing_version:
            warnings.warn(f"Data already exists as version {existing_version}")
            return existing_version
        
        # Store data
        file_path = self._store_data(data, version_id)
        size_bytes = self._get_file_size(file_path)
        
        # Create version record
        version = DataVersion(
            version_id=version_id,
            parent_version=parent_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=created_by,
            description=description,
            data_hash=data_hash,
            file_path=str(file_path.relative_to(self.storage_dir)),
            size_bytes=size_bytes,
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Store in database
        self._store_version(version)
        
        # Store metadata
        self._store_metadata(version_id, version)
        
        return version_id
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate hash of data"""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            # For DataFrames, hash the values and column names
            content = f"{data.columns.tolist()}{data.values.tobytes()}"
            return hashlib.sha256(content.encode()).hexdigest()
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            # For numpy arrays
            return hashlib.sha256(data.tobytes()).hexdigest()
        elif isinstance(data, (dict, list)):
            # For JSON-serializable objects
            content = json.dumps(data, sort_keys=True)
            return hashlib.sha256(content.encode()).hexdigest()
        elif isinstance(data, str):
            # For strings
            return hashlib.sha256(data.encode()).hexdigest()
        else:
            # For other objects, use pickle
            try:
                content = pickle.dumps(data)
                return hashlib.sha256(content).hexdigest()
            except Exception:
                # Fallback to string representation
                content = str(data)
                return hashlib.sha256(content.encode()).hexdigest()
    
    def _find_version_by_hash(self, data_hash: str) -> Optional[str]:
        """Find existing version with same hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT version_id FROM data_versions WHERE data_hash = ?",
                (data_hash,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def _store_data(self, data: Any, version_id: str) -> Path:
        """Store data to file system"""
        file_path = self.data_dir / f"{version_id}.pkl"
        
        try:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                # Store as both pickle and CSV for convenience
                data.to_pickle(file_path)
                csv_path = self.data_dir / f"{version_id}.csv"
                data.to_csv(csv_path, index=False)
            elif isinstance(data, (dict, list)):
                # Store as JSON
                json_path = self.data_dir / f"{version_id}.json"
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                file_path = json_path
            else:
                # Store as pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
        except Exception as e:
            warnings.warn(f"Failed to store data: {e}")
            # Fallback to string storage
            text_path = self.data_dir / f"{version_id}.txt"
            with open(text_path, 'w') as f:
                f.write(str(data))
            file_path = text_path
        
        return file_path
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        try:
            return file_path.stat().st_size
        except:
            return 0
    
    def _store_version(self, version: DataVersion):
        """Store version in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_versions 
                (version_id, parent_version, created_at, created_by, description, 
                 data_hash, file_path, size_bytes, metadata, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.version_id,
                version.parent_version,
                version.created_at,
                version.created_by,
                version.description,
                version.data_hash,
                version.file_path,
                version.size_bytes,
                json.dumps(version.metadata),
                json.dumps(version.tags)
            ))
            conn.commit()
    
    def _store_metadata(self, version_id: str, version: DataVersion):
        """Store detailed metadata"""
        metadata_path = self.metadata_dir / f"{version_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(version), f, indent=2)
    
    def load_version(self, version_id: str) -> Tuple[Any, DataVersion]:
        """Load data and metadata for a version"""
        # Get version info
        version_info = self.get_version_info(version_id)
        if not version_info:
            raise ValueError(f"Version {version_id} not found")
        
        # Load data
        data_path = self.storage_dir / version_info.file_path
        
        try:
            if data_path.suffix == '.pkl':
                if PANDAS_AVAILABLE:
                    data = pd.read_pickle(data_path)
                else:
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
            elif data_path.suffix == '.json':
                with open(data_path, 'r') as f:
                    data = json.load(f)
            elif data_path.suffix == '.csv':
                if PANDAS_AVAILABLE:
                    data = pd.read_csv(data_path)
                else:
                    raise ValueError("Pandas required to load CSV files")
            else:
                with open(data_path, 'r') as f:
                    data = f.read()
        except Exception as e:
            raise ValueError(f"Failed to load data for version {version_id}: {e}")
        
        return data, version_info
    
    def get_version_info(self, version_id: str) -> Optional[DataVersion]:
        """Get version information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT version_id, parent_version, created_at, created_by, description,
                       data_hash, file_path, size_bytes, metadata, tags
                FROM data_versions WHERE version_id = ?
            """, (version_id,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            return DataVersion(
                version_id=result[0],
                parent_version=result[1],
                created_at=result[2],
                created_by=result[3],
                description=result[4],
                data_hash=result[5],
                file_path=result[6],
                size_bytes=result[7],
                metadata=json.loads(result[8] or '{}'),
                tags=json.loads(result[9] or '[]')
            )
    
    def list_versions(self, tag_filter: Optional[str] = None,
                     created_by: Optional[str] = None,
                     since: Optional[str] = None) -> List[DataVersion]:
        """List all versions with optional filtering"""
        query = "SELECT * FROM data_versions WHERE 1=1"
        params = []
        
        if created_by:
            query += " AND created_by = ?"
            params.append(created_by)
        
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            results = cursor.fetchall()
            
            versions = []
            for row in results:
                version = DataVersion(
                    version_id=row[0],
                    parent_version=row[1],
                    created_at=row[2],
                    created_by=row[3],
                    description=row[4],
                    data_hash=row[5],
                    file_path=row[6],
                    size_bytes=row[7],
                    metadata=json.loads(row[8] or '{}'),
                    tags=json.loads(row[9] or '[]')
                )
                
                # Apply tag filter
                if tag_filter and tag_filter not in version.tags:
                    continue
                
                versions.append(version)
            
            return versions
    
    def get_version_tree(self, root_version: Optional[str] = None) -> Dict[str, Any]:
        """Get version dependency tree"""
        versions = self.list_versions()
        
        # Build tree structure
        tree = {}
        children = {}
        
        for version in versions:
            version_id = version.version_id
            parent_id = version.parent_version
            
            # Initialize node
            tree[version_id] = {
                'version': version,
                'children': []
            }
            
            # Track parent-child relationships
            if parent_id:
                if parent_id not in children:
                    children[parent_id] = []
                children[parent_id].append(version_id)
        
        # Build children lists
        for parent_id, child_list in children.items():
            if parent_id in tree:
                tree[parent_id]['children'] = child_list
        
        # If root version specified, return subtree
        if root_version and root_version in tree:
            return {root_version: tree[root_version]}
        
        # Return roots (versions with no parents)
        roots = {vid: node for vid, node in tree.items() 
                if node['version'].parent_version is None}
        
        return roots if roots else tree
    
    def delete_version(self, version_id: str, force: bool = False):
        """Delete a version (with safety checks)"""
        # Check if version has children
        children = self._get_version_children(version_id)
        if children and not force:
            raise ValueError(f"Version {version_id} has children: {children}. Use force=True to delete anyway.")
        
        # Get version info for cleanup
        version_info = self.get_version_info(version_id)
        if not version_info:
            warnings.warn(f"Version {version_id} not found")
            return
        
        # Delete files
        try:
            data_path = self.storage_dir / version_info.file_path
            if data_path.exists():
                data_path.unlink()
            
            # Delete related files
            data_stem = data_path.stem
            for related_file in self.data_dir.glob(f"{data_stem}.*"):
                related_file.unlink()
            
            # Delete metadata
            metadata_path = self.metadata_dir / f"{version_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()
                
        except Exception as e:
            warnings.warn(f"Error deleting files for version {version_id}: {e}")
        
        # Delete database records
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM processing_steps WHERE output_version = ?", (version_id,))
            conn.execute("DELETE FROM data_versions WHERE version_id = ?", (version_id,))
            conn.commit()
    
    def _get_version_children(self, version_id: str) -> List[str]:
        """Get child versions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT version_id FROM data_versions WHERE parent_version = ?",
                (version_id,)
            )
            return [row[0] for row in cursor.fetchall()]


class LineageTracker:
    """Tracks data lineage and processing history"""
    
    def __init__(self, storage_dir: str = "data_lineage"):
        """Initialize lineage tracker"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.storage_dir / "lineage.db"
        self._init_database()
        
        # Git integration
        self.git_repo = self._init_git_tracking()
        
    def _init_database(self):
        """Initialize lineage database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    properties TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_node TEXT NOT NULL,
                    target_node TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    properties TEXT,
                    FOREIGN KEY (source_node) REFERENCES lineage_nodes (node_id),
                    FOREIGN KEY (target_node) REFERENCES lineage_nodes (node_id)
                )
            """)
            
            conn.commit()
    
    def _init_git_tracking(self):
        """Initialize git repository for code versioning"""
        if not GIT_AVAILABLE:
            return None
        
        try:
            # Try to find existing git repo
            repo_path = Path.cwd()
            while repo_path.parent != repo_path:
                if (repo_path / '.git').exists():
                    return git.Repo(repo_path)
                repo_path = repo_path.parent
            
            # No git repo found
            return None
        except Exception:
            return None
    
    def record_processing_step(self, step_type: str, function_name: str,
                             parameters: Dict[str, Any],
                             input_data_versions: List[str],
                             output_data_version: str,
                             execution_time: float = 0.0) -> str:
        """Record a data processing step"""
        step_id = f"step_{uuid.uuid4().hex[:12]}"
        
        # Get git commit if available
        git_commit = None
        if self.git_repo:
            try:
                git_commit = self.git_repo.head.commit.hexsha
            except:
                pass
        
        # Collect environment info
        environment_info = {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'platform': os.sys.platform,
            'working_directory': str(Path.cwd()),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Create processing step
        step = ProcessingStep(
            step_id=step_id,
            step_type=step_type,
            function_name=function_name,
            parameters=parameters,
            input_versions=input_data_versions,
            output_version=output_data_version,
            execution_time=execution_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit,
            environment_info=environment_info
        )
        
        # Store in database
        self._store_processing_step(step)
        
        # Create lineage nodes and edges
        self._create_lineage_nodes_and_edges(step)
        
        return step_id
    
    def _store_processing_step(self, step: ProcessingStep):
        """Store processing step in database"""
        # First store in version manager's database
        version_manager = DataVersionManager()
        with sqlite3.connect(version_manager.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processing_steps
                (step_id, step_type, function_name, parameters, input_versions,
                 output_version, execution_time, timestamp, git_commit, environment_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                step.step_id,
                step.step_type,
                step.function_name,
                json.dumps(step.parameters),
                json.dumps(step.input_versions),
                step.output_version,
                step.execution_time,
                step.timestamp,
                step.git_commit,
                json.dumps(step.environment_info)
            ))
            conn.commit()
    
    def _create_lineage_nodes_and_edges(self, step: ProcessingStep):
        """Create lineage graph nodes and edges"""
        with sqlite3.connect(self.db_path) as conn:
            # Create process node
            process_node = LineageNode(
                node_id=step.step_id,
                node_type='process',
                name=step.function_name,
                description=f"{step.step_type} processing step",
                properties={
                    'parameters': step.parameters,
                    'execution_time': step.execution_time,
                    'git_commit': step.git_commit
                }
            )
            
            conn.execute("""
                INSERT OR REPLACE INTO lineage_nodes
                (node_id, node_type, name, description, properties, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                process_node.node_id,
                process_node.node_type,
                process_node.name,
                process_node.description,
                json.dumps(process_node.properties),
                process_node.created_at
            ))
            
            # Create edges from inputs to process
            for input_version in step.input_versions:
                edge = LineageEdge(
                    edge_id=f"edge_{uuid.uuid4().hex[:8]}",
                    source_node=input_version,
                    target_node=step.step_id,
                    edge_type='input'
                )
                
                conn.execute("""
                    INSERT OR REPLACE INTO lineage_edges
                    (edge_id, source_node, target_node, edge_type, properties)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    edge.edge_id,
                    edge.source_node,
                    edge.target_node,
                    edge.edge_type,
                    json.dumps(edge.properties)
                ))
            
            # Create edge from process to output
            edge = LineageEdge(
                edge_id=f"edge_{uuid.uuid4().hex[:8]}",
                source_node=step.step_id,
                target_node=step.output_version,
                edge_type='output'
            )
            
            conn.execute("""
                INSERT OR REPLACE INTO lineage_edges
                (edge_id, source_node, target_node, edge_type, properties)
                VALUES (?, ?, ?, ?, ?)
            """, (
                edge.edge_id,
                edge.source_node,
                edge.target_node,
                edge.edge_type,
                json.dumps(edge.properties)
            ))
            
            conn.commit()
    
    def get_lineage_graph(self, node_id: str, depth: int = 3) -> Dict[str, Any]:
        """Get lineage graph around a specific node"""
        visited_nodes = set()
        visited_edges = set()
        
        def traverse(current_node: str, current_depth: int, direction: str = 'both'):
            if current_depth <= 0 or current_node in visited_nodes:
                return
            
            visited_nodes.add(current_node)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get outgoing edges (downstream)
                if direction in ['both', 'downstream']:
                    cursor = conn.execute("""
                        SELECT edge_id, target_node, edge_type, properties
                        FROM lineage_edges WHERE source_node = ?
                    """, (current_node,))
                    
                    for edge_id, target_node, edge_type, properties in cursor.fetchall():
                        if edge_id not in visited_edges:
                            visited_edges.add(edge_id)
                            traverse(target_node, current_depth - 1, direction)
                
                # Get incoming edges (upstream)
                if direction in ['both', 'upstream']:
                    cursor = conn.execute("""
                        SELECT edge_id, source_node, edge_type, properties
                        FROM lineage_edges WHERE target_node = ?
                    """, (current_node,))
                    
                    for edge_id, source_node, edge_type, properties in cursor.fetchall():
                        if edge_id not in visited_edges:
                            visited_edges.add(edge_id)
                            traverse(source_node, current_depth - 1, direction)
        
        # Start traversal
        traverse(node_id, depth)
        
        # Collect node and edge details
        nodes = {}
        edges = {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Get node details
            for node_id in visited_nodes:
                cursor = conn.execute("""
                    SELECT node_id, node_type, name, description, properties, created_at
                    FROM lineage_nodes WHERE node_id = ?
                """, (node_id,))
                
                result = cursor.fetchone()
                if result:
                    nodes[node_id] = {
                        'node_id': result[0],
                        'node_type': result[1],
                        'name': result[2],
                        'description': result[3],
                        'properties': json.loads(result[4] or '{}'),
                        'created_at': result[5]
                    }
            
            # Get edge details
            for edge_id in visited_edges:
                cursor = conn.execute("""
                    SELECT edge_id, source_node, target_node, edge_type, properties
                    FROM lineage_edges WHERE edge_id = ?
                """, (edge_id,))
                
                result = cursor.fetchone()
                if result:
                    edges[edge_id] = {
                        'edge_id': result[0],
                        'source_node': result[1],
                        'target_node': result[2],
                        'edge_type': result[3],
                        'properties': json.loads(result[4] or '{}')
                    }
        
        return {
            'nodes': nodes,
            'edges': edges,
            'center_node': node_id,
            'depth': depth
        }
    
    def export_lineage_graph(self, output_path: str, format: str = 'graphviz') -> str:
        """Export complete lineage graph"""
        if format == 'graphviz':
            return self._export_graphviz(output_path)
        elif format == 'json':
            return self._export_json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_graphviz(self, output_path: str) -> str:
        """Export lineage graph as Graphviz DOT file"""
        dot_content = ["digraph lineage {"]
        dot_content.append("  rankdir=LR;")
        dot_content.append("  node [shape=box];")
        
        with sqlite3.connect(self.db_path) as conn:
            # Add nodes
            cursor = conn.execute("SELECT node_id, node_type, name FROM lineage_nodes")
            for node_id, node_type, name in cursor.fetchall():
                color = {
                    'data': 'lightblue',
                    'process': 'lightgreen',
                    'parameter': 'lightyellow'
                }.get(node_type, 'white')
                
                dot_content.append(f'  "{node_id}" [label="{name}\\n({node_type})" fillcolor="{color}" style=filled];')
            
            # Add edges
            cursor = conn.execute("SELECT source_node, target_node, edge_type FROM lineage_edges")
            for source, target, edge_type in cursor.fetchall():
                style = {
                    'input': 'solid',
                    'output': 'solid',
                    'parameter': 'dashed'
                }.get(edge_type, 'solid')
                
                dot_content.append(f'  "{source}" -> "{target}" [style="{style}" label="{edge_type}"];')
        
        dot_content.append("}")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(dot_content))
        
        return output_path
    
    def _export_json(self, output_path: str) -> str:
        """Export lineage graph as JSON"""
        graph_data = {'nodes': [], 'edges': []}
        
        with sqlite3.connect(self.db_path) as conn:
            # Export nodes
            cursor = conn.execute("SELECT * FROM lineage_nodes")
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                node_data = dict(zip(columns, row))
                node_data['properties'] = json.loads(node_data['properties'] or '{}')
                graph_data['nodes'].append(node_data)
            
            # Export edges
            cursor = conn.execute("SELECT * FROM lineage_edges")
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                edge_data = dict(zip(columns, row))
                edge_data['properties'] = json.loads(edge_data['properties'] or '{}')
                graph_data['edges'].append(edge_data)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        return output_path


class VersionedDataset:
    """A dataset with built-in versioning and lineage tracking"""
    
    def __init__(self, data: Any = None, description: str = "Versioned dataset",
                 version_manager: Optional[DataVersionManager] = None,
                 lineage_tracker: Optional[LineageTracker] = None):
        """Initialize versioned dataset"""
        self.version_manager = version_manager or DataVersionManager()
        self.lineage_tracker = lineage_tracker or LineageTracker()
        
        self.current_version: Optional[str] = None
        self.current_data: Any = None
        
        if data is not None:
            self.create_version(data, description)
    
    def create_version(self, data: Any, description: str,
                      created_by: str = "system",
                      tags: Optional[List[str]] = None) -> str:
        """Create new version of the dataset"""
        version_id = self.version_manager.create_version(
            data, description, self.current_version, created_by, tags
        )
        
        self.current_version = version_id
        self.current_data = data
        
        return version_id
    
    def apply_transformation(self, transform_func: Callable, 
                           description: str,
                           **kwargs) -> 'VersionedDataset':
        """Apply transformation and create new version"""
        import time
        start_time = time.time()
        
        # Apply transformation
        new_data = transform_func(self.current_data, **kwargs)
        
        # Create new version
        new_version_id = self.create_version(new_data, description)
        
        # Record lineage
        execution_time = time.time() - start_time
        self.lineage_tracker.record_processing_step(
            step_type='transformation',
            function_name=transform_func.__name__,
            parameters=kwargs,
            input_data_versions=[self.current_version] if self.current_version else [],
            output_data_version=new_version_id,
            execution_time=execution_time
        )
        
        # Create new dataset instance
        new_dataset = VersionedDataset(
            version_manager=self.version_manager,
            lineage_tracker=self.lineage_tracker
        )
        new_dataset.current_version = new_version_id
        new_dataset.current_data = new_data
        
        return new_dataset
    
    def checkout_version(self, version_id: str):
        """Switch to a specific version"""
        data, version_info = self.version_manager.load_version(version_id)
        self.current_version = version_id
        self.current_data = data
        return self
    
    def get_lineage(self, depth: int = 3) -> Dict[str, Any]:
        """Get lineage for current version"""
        if not self.current_version:
            return {'nodes': {}, 'edges': {}}
        
        return self.lineage_tracker.get_lineage_graph(self.current_version, depth)
    
    def export_provenance_report(self, output_path: str = "provenance_report.html"):
        """Export comprehensive provenance report"""
        if not self.current_version:
            print("No current version to generate report for")
            return
        
        # Get version info
        version_info = self.version_manager.get_version_info(self.current_version)
        
        # Get lineage
        lineage = self.get_lineage(depth=5)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Provenance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .version-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .lineage-section {{ margin: 20px 0; }}
                .node {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
                .data-node {{ background-color: #e3f2fd; }}
                .process-node {{ background-color: #e8f5e8; }}
                .parameter-node {{ background-color: #fff3e0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>📊 Data Provenance Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="version-info">
                <h2>Current Version Information</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Version ID</td><td>{version_info.version_id}</td></tr>
                    <tr><td>Description</td><td>{version_info.description}</td></tr>
                    <tr><td>Created At</td><td>{version_info.created_at}</td></tr>
                    <tr><td>Created By</td><td>{version_info.created_by}</td></tr>
                    <tr><td>Data Hash</td><td>{version_info.data_hash[:16]}...</td></tr>
                    <tr><td>Size</td><td>{version_info.size_bytes:,} bytes</td></tr>
                    <tr><td>Tags</td><td>{', '.join(version_info.tags) if version_info.tags else 'None'}</td></tr>
                </table>
            </div>
            
            <div class="lineage-section">
                <h2>Data Lineage</h2>
                <p>Showing {len(lineage['nodes'])} nodes and {len(lineage['edges'])} relationships.</p>
        """
        
        # Add lineage nodes
        for node_id, node_data in lineage['nodes'].items():
            node_class = f"{node_data['node_type']}-node"
            html_content += f"""
                <div class="node {node_class}">
                    <strong>{node_data['name']}</strong> ({node_data['node_type']})
                    <br><small>ID: {node_id}</small>
                    <br><small>Created: {node_data['created_at']}</small>
                    {f"<br><small>Description: {node_data['description']}</small>" if node_data['description'] else ""}
                </div>
            """
        
        html_content += """
            </div>
            
            <footer style="margin-top: 40px; text-align: center; color: #6c757d;">
                <p>Generated by CsPbBr₃ Digital Twin Data Versioning System</p>
            </footer>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Provenance report saved to: {output_path}")


def versioned_function(description: str = "", tags: Optional[List[str]] = None):
    """Decorator to automatically version function outputs"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get version manager and lineage tracker
            version_manager = DataVersionManager()
            lineage_tracker = LineageTracker()
            
            # Execute function
            import time
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Create version
            func_description = description or f"Output from {func.__name__}"
            version_id = version_manager.create_version(
                result, func_description, tags=tags
            )
            
            # Record lineage
            # Note: This is a simplified approach - in practice, you'd want to
            # track input data versions as well
            lineage_tracker.record_processing_step(
                step_type='function_call',
                function_name=func.__name__,
                parameters=kwargs,
                input_data_versions=[],  # Would need to be tracked
                output_data_version=version_id,
                execution_time=execution_time
            )
            
            print(f"📦 Function output versioned as: {version_id}")
            return result, version_id
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demonstration of data versioning and lineage tracking
    print("📚 Data Versioning and Lineage Tracking Demo")
    
    # Initialize components
    version_manager = DataVersionManager()
    lineage_tracker = LineageTracker()
    
    print("🗂️ Testing data versioning...")
    
    # Create some sample data
    if PANDAS_AVAILABLE:
        sample_data = pd.DataFrame({
            'temperature': [150, 160, 140],
            'concentration': [1.0, 1.2, 0.8],
            'plqy': [0.85, 0.78, 0.92]
        })
    else:
        sample_data = {
            'temperature': [150, 160, 140],
            'concentration': [1.0, 1.2, 0.8],
            'plqy': [0.85, 0.78, 0.92]
        }
    
    # Create first version
    v1 = version_manager.create_version(
        sample_data, 
        "Initial experimental data",
        created_by="researcher_1",
        tags=["experimental", "initial"]
    )
    print(f"   Created version: {v1}")
    
    # Create modified data and second version
    if PANDAS_AVAILABLE:
        modified_data = sample_data.copy()
        modified_data['plqy'] = modified_data['plqy'] * 1.1  # 10% increase
    else:
        modified_data = sample_data.copy()
        modified_data['plqy'] = [x * 1.1 for x in modified_data['plqy']]
    
    v2 = version_manager.create_version(
        modified_data,
        "Data with calibration correction",
        parent_version=v1,
        created_by="researcher_2",
        tags=["calibrated", "corrected"]
    )
    print(f"   Created version: {v2}")
    
    # Record processing step
    step_id = lineage_tracker.record_processing_step(
        step_type="calibration",
        function_name="apply_calibration_correction",
        parameters={"correction_factor": 1.1},
        input_data_versions=[v1],
        output_data_version=v2,
        execution_time=0.5
    )
    print(f"   Recorded processing step: {step_id}")
    
    # Test versioned dataset
    print("\n📊 Testing versioned dataset...")
    
    @versioned_function("Processed data with noise reduction", ["processed", "cleaned"])
    def reduce_noise(data):
        """Mock noise reduction function"""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            # Add small random improvement to PLQY
            result = data.copy()
            result['plqy'] = result['plqy'] * 1.05
            return result
        else:
            result = data.copy()
            result['plqy'] = [x * 1.05 for x in result['plqy']]
            return result
    
    # Apply versioned function
    processed_data, processed_version = reduce_noise(modified_data)
    print(f"   Function output versioned as: {processed_version}")
    
    # Test versioned dataset workflow
    print("\n🔄 Testing versioned dataset workflow...")
    dataset = VersionedDataset(sample_data, "Initial dataset")
    
    # Apply transformation
    def normalize_temperature(data):
        """Normalize temperature to 0-1 range"""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            result = data.copy()
            temp_min = result['temperature'].min()
            temp_max = result['temperature'].max()
            result['temperature_normalized'] = (result['temperature'] - temp_min) / (temp_max - temp_min)
            return result
        else:
            result = data.copy()
            temps = result['temperature']
            temp_min = min(temps)
            temp_max = max(temps)
            result['temperature_normalized'] = [(t - temp_min) / (temp_max - temp_min) for t in temps]
            return result
    
    normalized_dataset = dataset.apply_transformation(
        normalize_temperature,
        "Added normalized temperature feature"
    )
    
    print(f"   Applied transformation, new version: {normalized_dataset.current_version}")
    
    # List all versions
    print("\n📋 Listing all versions...")
    versions = version_manager.list_versions()
    for version in versions[:5]:  # Show first 5
        print(f"   {version.version_id}: {version.description} (by {version.created_by})")
    
    # Get lineage
    print("\n🌐 Getting lineage information...")
    lineage = normalized_dataset.get_lineage()
    print(f"   Lineage contains {len(lineage['nodes'])} nodes and {len(lineage['edges'])} edges")
    
    # Export reports
    print("\n📄 Generating reports...")
    
    # Export lineage graph
    lineage_tracker.export_lineage_graph("lineage_graph.dot", "graphviz")
    lineage_tracker.export_lineage_graph("lineage_graph.json", "json")
    
    # Export provenance report
    normalized_dataset.export_provenance_report("provenance_report.html")
    
    # Get version tree
    print("\n🌳 Version tree structure:")
    tree = version_manager.get_version_tree()
    
    def print_tree(tree_dict, indent=0):
        for version_id, node in tree_dict.items():
            print("  " * indent + f"├── {version_id}: {node['version'].description}")
            if node['children']:
                child_tree = {child: tree.get(child, {'version': version_manager.get_version_info(child), 'children': []}) 
                             for child in node['children']}
                print_tree(child_tree, indent + 1)
    
    print_tree(tree)
    
    print("\n✅ Data versioning and lineage tracking demo complete!")
    print("📁 Generated files:")
    print("   - lineage_graph.dot")
    print("   - lineage_graph.json") 
    print("   - provenance_report.html")
    print("   - Version database: data_versions/versions.db")
    print("   - Lineage database: data_lineage/lineage.db")
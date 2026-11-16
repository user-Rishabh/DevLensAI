import os
import json
from .parsers.js_parser import parse_js_ts_imports
from .parsers.python_parser import parse_python_imports

def parse_ipynb_imports(content: str, current_file: str, project_files: set[str]) -> list[str]:
    """
    Extracts and parses Python import statements from code cells within Jupyter Notebooks (.ipynb).
    """
    try:
        nb_data = json.loads(content)
        code_lines = []
        for cell in nb_data.get("cells", []):
            if cell.get("cell_type") == "code":
                src = cell.get("source", [])
                if isinstance(src, list):
                    code_lines.extend(src)
                elif isinstance(src, str):
                    code_lines.append(src)
        if code_lines:
            joined_code = "".join(code_lines)
            return parse_python_imports(joined_code, current_file, project_files)
    except Exception:
        pass
    return []

def get_all_project_files(file_tree: dict) -> set[str]:
    """
    Traverses the file tree dictionary and returns a set of all relative file paths.
    """
    files = set()
    def traverse(node):
        if not node:
            return
        if node.get("type") == "file":
            path = node.get("path")
            if path:
                files.add(path)
        elif node.get("type") == "folder":
            for child in node.get("children", []):
                traverse(child)
    traverse(file_tree)
    return files

def extract_dependencies(local_path: str, file_tree: dict) -> list[dict]:
    """
    Analyzes Python, JS, and TS files to extract internal file-to-file import edges.
    Returns a list of edges: [{"from": "file_a", "to": "file_b"}]
    """
    project_files = get_all_project_files(file_tree)
    edges = []
    
    for file_rel_path in project_files:
        full_path = os.path.join(local_path, file_rel_path)
        if not os.path.exists(full_path):
            continue
            
        # Determine language parser
        _, ext = os.path.splitext(file_rel_path)
        ext = ext.lower()
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except OSError:
            continue
            
        imported_files = []
        if ext in ['.js', '.jsx', '.ts', '.tsx']:
            imported_files = parse_js_ts_imports(content, file_rel_path, project_files)
        elif ext == '.py':
            imported_files = parse_python_imports(content, file_rel_path, project_files)
        elif ext == '.ipynb':
            imported_files = parse_ipynb_imports(content, file_rel_path, project_files)
            
        for target in imported_files:
            # Avoid self-dependencies
            if target != file_rel_path:
                edges.append({
                    "from": file_rel_path,
                    "to": target
                })
                
    return edges

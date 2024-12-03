import os
import ast
import pkg_resources
import sys
from pathlib import Path

def find_imports(file_path):
    """Extract all import statements from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError:
            print(f"Syntax error in {file_path}, skipping...")
            return set()
    
    imports = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.add(node.module.split('.')[0])
    
    return imports

def get_installed_packages():
    """Get a mapping of package names to their versions."""
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def is_standard_library(module_name):
    """Check if a module is part of the Python standard library."""
    if module_name in sys.stdlib_module_names:
        return True
    try:
        module_path = __import__(module_name).__file__
        return 'site-packages' not in module_path
    except (ImportError, AttributeError):
        return False

def main():
    # Get Python files recursively from current directory
    python_files = list(Path('.').rglob('*.py'))
    
    if not python_files:
        print("No Python files found in the current directory!")
        return
    
    all_imports = set()
    for file_path in python_files:
        imports = find_imports(str(file_path))
        all_imports.update(imports)
    
    # Filter out standard library modules
    third_party_imports = {imp for imp in all_imports if not is_standard_library(imp)}
    
    # Get installed versions
    installed_packages = get_installed_packages()
    
    # Create requirements.txt
    with open('requirements.txt', 'w') as f:
        for package in sorted(third_party_imports):
            if package in installed_packages:
                f.write(f'{package}=={installed_packages[package]}\n')
            else:
                print(f"Warning: Package '{package}' is imported but not installed")

if __name__ == '__main__':
    main()
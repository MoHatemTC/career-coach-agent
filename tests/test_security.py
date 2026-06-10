import os
import glob
import pytest

def test_no_hardcoded_secrets_in_repo():
    """
    Scans the repository's .py and .md files to ensure no hardcoded secrets exist.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define secret patterns to look for
    secret_patterns = [
        "LITELLM_API_KEY=",
        "sk-ant-",
        "sk-proj-",
        "sk-or-"
    ]
    
    # Collect all .py and .md files, ignoring virtual environments and .git
    files_to_check = []
    for root, dirs, files in os.walk(project_root):
        # Exclude directories that shouldn't be scanned
        dirs[:] = [d for d in dirs if d not in ('.git', 'venv', 'env', '.pytest_cache', '__pycache__')]
        for file in files:
            if file.endswith('.py') or file.endswith('.md'):
                files_to_check.append(os.path.join(root, file))
                
    for filepath in files_to_check:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            for pattern in secret_patterns:
                assert pattern not in content, f"Hardcoded secret '{pattern}' found in {filepath}!"

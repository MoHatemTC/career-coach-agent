import os
import glob
import pytest

def test_no_hardcoded_secrets_in_repo():
    """
    Scans the repository's source and configuration files to ensure no hardcoded secrets exist.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define secret patterns to look for
    secret_patterns = [
        "LITELLM_" + "API_KEY=",
        "OPENAI_" + "API_KEY=",
        "GEMINI_" + "API_KEY=",
        "ANTHROPIC_" + "API_KEY=",
        "sk" + "-ant-",
        "sk" + "-proj-",
        "sk" + "-or-"
    ]
    
    # Collect relevant files, ignoring virtual environments and .git
    files_to_check = []
    for root, dirs, files in os.walk(project_root):
        # Exclude directories that shouldn't be scanned
        dirs[:] = [d for d in dirs if d not in ('.git', 'venv', 'env', '.pytest_cache', '__pycache__', 'tests')]
        for file in files:
            if file.endswith(('.py', '.md', '.yaml', '.yml', '.json', '.toml')):
                if file == '.pre-commit-config.yaml':
                    continue
                files_to_check.append(os.path.join(root, file))
                
    for filepath in files_to_check:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            for pattern in secret_patterns:
                assert pattern not in content, f"Hardcoded secret '{pattern}' found in {filepath}!"

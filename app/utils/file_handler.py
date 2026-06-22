import os
from typing import Tuple

MAX_FILE_SIZE_MB = 2
UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

def validate_cv(uploaded_file, max_mb: int = MAX_FILE_SIZE_MB) -> Tuple[bool, str]:
    """Validates the uploaded file for both size and specific file types."""
    if uploaded_file is None:
        return False, "No file uploaded."
    
    # 1. Type Validation
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type: {file_ext}. Only PDF and DOCX are allowed."

    # 2. Size Validation
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_mb:
        return False, f"File size ({file_size_mb:.2f} MB) exceeds the {max_mb} MB limit."
    
    return True, "Valid file."

def save_cv_file(uploaded_file, username: str) -> str:
    """Saves the valid CV file to the local uploads directory."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, f"{username}_cv_{uploaded_file.name}")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path
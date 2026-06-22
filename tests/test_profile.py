import pytest
import os
from app.db.users import save_user_profile, get_user_profile
from app.db.connection import init_db
from app.utils.file_handler import validate_cv
from app.schemas.profile import UserProfile
from pydantic import ValidationError

@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Fixture to securely bind DB_PATH for reproducible testing before imports evaluate."""
    test_db = "test_career_coach.db"
    monkeypatch.setenv("DB_PATH", test_db)
    init_db()
    yield
    # Cleanup after tests complete
    if os.path.exists(test_db):
        os.remove(test_db)

def test_sqlite_roundtrip_save_and_load():
    """Tests complete DB persistence layer without touching the production file."""
    test_username = "test_roundtrip_user"
    test_data = {
        "experience_years": "Student",
        "career_level": "Entry Level",
        "job_types": ["Full Time"],
        "workplace_settings": ["Remote"],
        "job_titles": ["AI Engineer", "Data Scientist"],
        "job_categories": ["IT"],
        "hide_minimum_salary": False,
        "let_companies_find_me": True,
        "make_profile_public": True,
        "cv_file_path": None
    }
    
    save_user_profile(test_username, test_data)
    loaded_data = get_user_profile(test_username)
    
    assert loaded_data["experience_years"] == "Student"
    assert "AI Engineer" in loaded_data["job_titles"]

class MockUploadedFile:
    def __init__(self, size_bytes, name):
        self.size = size_bytes
        self.name = name

def test_cv_upload_validation():
    """Tests both boundary size limits and file extension checks."""
    # Size limit violation
    large_file = MockUploadedFile(size_bytes=3 * 1024 * 1024, name="large_cv.pdf")
    is_valid, msg = validate_cv(large_file, max_mb=2)
    assert is_valid is False
    assert "exceeds" in msg

    # Type limit violation
    bad_type_file = MockUploadedFile(size_bytes=1 * 1024 * 1024, name="malicious_script.py")
    is_valid, msg = validate_cv(bad_type_file, max_mb=2)
    assert is_valid is False
    assert "Invalid file type" in msg

    # Valid execution
    valid_file = MockUploadedFile(size_bytes=1 * 1024 * 1024, name="good_cv.pdf")
    is_valid, msg = validate_cv(valid_file, max_mb=2)
    assert is_valid is True

def test_profile_schema_validation():
    """Tests Pydantic constraints, specifically the 10 job title maximum."""
    valid_data = {
        "experience_years": "Student",
        "career_level": "Entry Level",
        "job_types": ["Full Time"],
        "workplace_settings": ["Remote"],
        "job_titles": ["AI Engineer", "Data Scientist"],
        "job_categories": ["IT"]
    }
    
    # Should pass
    profile = UserProfile(**valid_data)
    assert len(profile.job_titles) == 2

    # Should fail boundary constraint
    invalid_data = valid_data.copy()
    invalid_data["job_titles"] = [f"Job {i}" for i in range(11)]
    
    with pytest.raises(ValidationError):
        UserProfile(**invalid_data)
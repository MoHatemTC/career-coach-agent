from pydantic import BaseModel, Field
from typing import List, Optional

class UserProfile(BaseModel):
    experience_years: str
    career_level: str
    job_types: List[str]
    workplace_settings: List[str]
    job_titles: List[str] = Field(..., max_length=10)
    job_categories: List[str]
    minimum_salary: Optional[int] = None
    hide_minimum_salary: bool = False
    let_companies_find_me: bool = True
    make_profile_public: bool = True
    cv_file_path: Optional[str] = None  
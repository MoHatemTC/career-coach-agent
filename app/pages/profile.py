import streamlit as st
from app.components.sidebar import render_sidebar
from app.schemas.profile import UserProfile
from app.db.users import save_user_profile, get_user_profile
from app.utils.file_handler import validate_cv, save_cv_file
from pydantic import ValidationError

st.set_page_config(page_title="Career Interests | Career Coach", layout="wide")
render_sidebar()

if 'username' not in st.session_state:
    st.warning("Please navigate to the Home page and select or create a user first.")
    st.stop()

CURRENT_USER = st.session_state['username']
existing_data = get_user_profile(CURRENT_USER)

st.title(f"Tell Us About Your Career Interests, {CURRENT_USER}")

with st.form("profile_form"):
    st.subheader("Experience & Level")
    exp_years = st.selectbox("How many years of experience do you have?", 
                             ["Less than 1 year", "1-3 years", "3-5 years", "5+ years"], 
                             index=None)
    
    career_level = st.radio("What is your current career level?", 
                            ["Student", "Entry Level", "Experienced", "Manager", "Senior Management", "Not specified"], 
                            index=None,
                            horizontal=True)
    
    st.subheader("Job Preferences")
    job_types = st.multiselect("What type(s) of job are you open to?", 
                               ["Full Time", "Part Time", "Freelance / Project", "Internship", "Shift Based", "Volunteering", "Student Activity"])
    
    workplace_settings = st.multiselect("What is your preferred workplace settings?", 
                                        ["On-site", "Remote", "Hybrid"])
    
    job_titles = st.multiselect("What are the job titles that describe what you are looking for? (Max 10)", 
                                ["AI Engineer", "Machine Learning Engineer", "Data Scientist", "Software Engineer"], 
                                max_selections=10)
    
    st.subheader("Industry & Compensation")
    job_categories = st.multiselect("What job categories are you interested in?", 
                                    ["IT/Software Development", "Engineering - Telecom/Technology", "Analyst/Research"])
    
    col1, col2 = st.columns(2)
    with col1:
        minimum_salary = st.number_input("What is the minimum salary you would accept? (EGP / Month)", min_value=0, step=1000, value=None)
    with col2:
        st.write("") 
        st.write("")
        hide_min_salary = st.checkbox("Hide my minimum salary from companies.")
        
    st.subheader("Visibility & Resumes")
    let_companies_find_me = st.toggle("Let companies find me on WUZZUF. (Recommended)", value=True)
    make_profile_public = st.toggle("Make my profile public. (Recommended)", value=True)
    
    st.divider()
    cv_file = st.file_uploader("Upload your CV (Max 2MB, PDF/DOCX)", type=["pdf", "docx"])
    
    submitted = st.form_submit_button("Save Profile Data")
    
    if submitted:
        saved_cv_path = existing_data.get("cv_file_path") 
        
        if cv_file is not None:
            # Updated function call here
            is_valid, msg = validate_cv(cv_file)
            if not is_valid:
                st.error(msg)
                st.stop() 
            else:
                saved_cv_path = save_cv_file(cv_file, CURRENT_USER)
                st.success(f"CV successfully saved to {saved_cv_path}")

        try:
            profile = UserProfile(
                experience_years=exp_years or "",
                career_level=career_level or "",
                job_types=job_types,
                workplace_settings=workplace_settings,
                job_titles=job_titles,
                job_categories=job_categories,
                minimum_salary=minimum_salary,
                hide_minimum_salary=hide_min_salary,
                let_companies_find_me=let_companies_find_me,
                make_profile_public=make_profile_public,
                cv_file_path=saved_cv_path
            )
            
            save_user_profile(CURRENT_USER, profile.model_dump())
            st.success("Your comprehensive career profile has been securely saved!")
            
        except ValidationError as e:
            st.error("Validation Error. Please check your inputs.")
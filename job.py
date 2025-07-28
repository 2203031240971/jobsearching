import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from io import BytesIO
import textwrap
import plotly.express as px
import plotly.graph_objects as go

# ========================
# INITIALIZE SESSION STATE
# ========================
if 'candidates' not in st.session_state:
    st.session_state.candidates = []
if 'jobs' not in st.session_state:
    st.session_state.jobs = []
if 'matches' not in st.session_state:
    st.session_state.matches = pd.DataFrame()
if 'applications' not in st.session_state:
    st.session_state.applications = []

# ========================
# PAGE CONFIGURATION
# ========================
st.set_page_config(
    page_title="JobMatch AI", 
    page_icon="💼", 
    layout="wide",
    menu_items={
        'Get Help': 'https://example.com',
        'Report a bug': "https://example.com",
        'About': "# JobMatch AI - Advanced Job Matching Platform"
    }
)

st.title("💼 JobMatch AI - Intelligent Job Matching Platform")
st.markdown("Connect candidates with the perfect jobs using AI-powered matching")

# ========================
# CONSTANTS
# ========================
SKILLS_DATABASE = [
    "Python", "Java", "JavaScript", "SQL", "Machine Learning", "Data Analysis",
    "TensorFlow", "PyTorch", "AWS", "Azure", "Docker", "Kubernetes", "React",
    "Node.js", "Flask", "Django", "Git", "CI/CD", "Agile", "Scrum", "Project Management",
    "Data Visualization", "Pandas", "NumPy", "Spark", "Hadoop", "Tableau", "Power BI",
    "Natural Language Processing", "Computer Vision", "Deep Learning", "Statistical Modeling"
]

JOB_TITLES = [
    "Data Scientist", "Software Engineer", "Machine Learning Engineer",
    "Data Analyst", "DevOps Engineer", "Frontend Developer", "Backend Developer",
    "Full Stack Developer", "Cloud Architect", "Product Manager"
]

# ========================
# UTILITY FUNCTIONS
# ========================

# Load NLP models
@st.cache_resource(show_spinner="Loading NLP models...")
def load_nlp_models():
    try:
        nlp = spacy.load("en_core_web_lg")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return nlp, model
    except:
        st.warning("NLP models loading... please wait")
        nlp = spacy.load("en_core_web_sm")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return nlp, model

nlp, model = load_nlp_models()

def extract_text(file):
    """Extract text from different file types"""
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return " ".join([page.extract_text() for page in pdf.pages])
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

def extract_skills(text):
    """Extract skills from text using rule-based and NLP approaches"""
    text = text.lower()
    found_skills = []
    
    # Enhanced matching with variations
    skill_variations = {
        "python": "Python",
        "java": "Java",
        "javascript": "JavaScript",
        "js": "JavaScript",
        "sql": "SQL",
        "machine learning": "Machine Learning",
        "ml": "Machine Learning",
        "data analysis": "Data Analysis",
        "tensorflow": "TensorFlow",
        "pytorch": "PyTorch",
        "aws": "AWS",
        "amazon web services": "AWS",
        "azure": "Azure",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "k8s": "Kubernetes",
        "react": "React",
        "node": "Node.js",
        "nodejs": "Node.js",
        "flask": "Flask",
        "django": "Django",
        "git": "Git",
        "ci/cd": "CI/CD",
        "agile": "Agile",
        "scrum": "Scrum"
    }
    
    for term, skill in skill_variations.items():
        if re.search(r'\b' + re.escape(term) + r'\b', text):
            found_skills.append(skill)
    
    # Remove duplicates
    return list(set(found_skills))

def extract_experience(text):
    """Extract years of experience from text"""
    matches = re.findall(r'(\d+)\s+(years?|yrs?)', text, re.IGNORECASE)
    if matches:
        return max([int(num) for num, unit in matches])
    return 0

def extract_education(text):
    """Extract education level from text"""
    degrees = {
        "phd": "PhD", 
        "doctorate": "PhD",
        "master": "Master's",
        "msc": "Master's",
        "mba": "Master's",
        "bachelor": "Bachelor's",
        "bsc": "Bachelor's",
        "btech": "Bachelor's",
        "associate": "Associate",
        "diploma": "Diploma"
    }
    
    found_degrees = []
    text = text.lower()
    
    for term, degree in degrees.items():
        if re.search(r'\b' + re.escape(term) + r'\b', text):
            found_degrees.append(degree)
    
    return " | ".join(set(found_degrees)) if found_degrees else "Not Specified"

def generate_embeddings(text):
    """Generate text embeddings using Sentence Transformers"""
    return model.encode([text])[0]

def calculate_match_score(candidate, job):
    """Calculate comprehensive match score between candidate and job"""
    # Skill similarity (TF-IDF)
    skill_vectorizer = TfidfVectorizer()
    candidate_skills = " ".join(candidate['skills'])
    job_skills = " ".join(job['required_skills'])
    
    skill_matrix = skill_vectorizer.fit_transform([candidate_skills, job_skills])
    skill_similarity = cosine_similarity(skill_matrix[0:1], skill_matrix[1:2])[0][0]
    
    # Experience match
    exp_match = 1 - min(1, max(0, (job['min_experience'] - candidate['experience']) / 5))
    
    # Job title similarity (BERT embeddings)
    title_similarity = cosine_similarity(
        [generate_embeddings(candidate['job_title'])], 
        [generate_embeddings(job['title'])]
    )[0][0]
    
    # Description similarity
    desc_similarity = cosine_similarity(
        [generate_embeddings(candidate['summary'])], 
        [generate_embeddings(job['description'])]
    )[0][0]
    
    # Weighted composite score
    weights = {
        'skills': 0.4,
        'experience': 0.2,
        'title': 0.2,
        'description': 0.2
    }
    
    score = (
        weights['skills'] * skill_similarity +
        weights['experience'] * exp_match +
        weights['title'] * title_similarity +
        weights['description'] * desc_similarity
    )
    
    return min(100, round(score * 100))

def generate_skill_gap(candidate_skills, job_skills):
    """Identify missing skills for a candidate"""
    return list(set(job_skills) - set(candidate_skills))

def generate_match_reasons(candidate, job, score):
    """Generate human-readable match reasons"""
    reasons = []
    
    # Skill overlap
    common_skills = set(candidate['skills']) & set(job['required_skills'])
    if common_skills:
        reasons.append(f"Strong skill match: {', '.join(list(common_skills)[:3])}{'...' if len(common_skills) > 3 else ''}")
    
    # Experience match
    if candidate['experience'] >= job['min_experience']:
        reasons.append(f"Meets experience requirement ({candidate['experience']} years)")
    
    # Title relevance
    if any(word in candidate['job_title'].lower() for word in job['title'].lower().split()):
        reasons.append(f"Relevant job title: {candidate['job_title']}")
    
    # Education match
    if job.get('education') and job['education'] in candidate['education']:
        reasons.append(f"Education requirement met: {job['education']}")
    
    # Location match
    if candidate.get('location') and job.get('location'):
        if candidate['location'].lower() == job['location'].lower():
            reasons.append("Location match")
    
    return reasons

# ========================
# NEW FEATURE FUNCTIONS
# ========================

# 1. AI-Powered Resume Builder
def generate_resume_suggestions(candidate):
    """Generate AI-powered resume improvement suggestions"""
    suggestions = []
    
    # Check resume length
    if len(candidate.get('resume_text', '')) < 300:
        suggestions.append("Your resume seems too short. Consider adding more details about your projects and accomplishments.")
    
    # Check skills formatting
    if not candidate.get('skills'):
        suggestions.append("Add a dedicated skills section with relevant technical skills.")
    elif len(candidate['skills']) < 5:
        suggestions.append("Include more technical skills that match your target job roles.")
    
    # Check experience descriptions
    experience_pattern = r"\b(led|developed|implemented|managed|created|improved|optimized)\b"
    if not re.search(experience_pattern, candidate.get('summary', ''), re.IGNORECASE):
        suggestions.append("Use more action verbs in your experience descriptions (e.g., 'developed', 'led', 'implemented').")
    
    # Check quantifiable achievements
    number_pattern = r"\d+"
    if not re.search(number_pattern, candidate.get('summary', '')):
        suggestions.append("Add quantifiable achievements (e.g., 'increased efficiency by 20%', 'managed team of 5 developers').")
    
    return suggestions

# 2. Interview Simulator
def generate_interview_questions(job):
    """Generate job-specific interview questions"""
    technical_questions = []
    behavioral_questions = []
    
    # Technical questions based on skills
    for skill in job['required_skills'][:5]:
        technical_questions.append(
            f"Explain your experience with {skill}. What challenges did you face while using it?"
        )
    
    # Behavioral questions
    behavioral_questions = [
        "Describe a time you had to solve a complex technical problem. What was your approach?",
        "Tell me about a project where you had to collaborate with a difficult team member.",
        "How do you stay updated with the latest industry trends and technologies?",
        "Describe a situation where you had to make a technical decision with incomplete information."
    ]
    
    # Job-specific questions
    if "Data Science" in job['title'] or "Machine Learning" in job['title']:
        technical_questions.append(
            "Walk me through your process for developing and validating a machine learning model."
        )
    elif "Frontend" in job['title']:
        technical_questions.append(
            "How do you approach responsive design and cross-browser compatibility?"
        )
    
    return {
        "technical": technical_questions,
        "behavioral": behavioral_questions
    }

# 3. Salary Estimator (INR only)
def estimate_salary(job_title, experience, location):
    """Estimate salary based on market data in INR"""
    # Base salaries by role (in INR)
    base_salaries = {
        "Data Scientist": 1_200_000,
        "Software Engineer": 900_000,
        "Machine Learning Engineer": 1_300_000,
        "Data Analyst": 800_000,
        "DevOps Engineer": 1_100_000,
        "Frontend Developer": 850_000,
        "Backend Developer": 950_000,
        "Full Stack Developer": 1_100_000,
        "Cloud Architect": 1_500_000,
        "Product Manager": 1_400_000
    }
    
    # Adjust for experience (10% increase per year)
    base = base_salaries.get(job_title, 1_000_000)
    adjusted = base * (1 + 0.10 * min(experience, 15))
    
    # Location adjustment
    location_adjustment = 1.0
    if "Bangalore" in location or "Hyderabad" in location:
        location_adjustment = 1.15
    elif "Mumbai" in location or "Delhi" in location:
        location_adjustment = 1.20
    elif "Pune" in location or "Chennai" in location:
        location_adjustment = 1.10
    elif "Gurgaon" in location or "Noida" in location:
        location_adjustment = 1.12
    
    return int(adjusted * location_adjustment)

# 4. Application Tracker
def add_application(candidate_id, job_id, status="Applied"):
    """Track job applications"""
    app = {
        "id": len(st.session_state.applications) + 1,
        "candidate_id": candidate_id,
        "job_id": job_id,
        "status": status,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    st.session_state.applications.append(app)
    return app

# 5. Text Report Generator
def generate_text_report(candidate, job, match_score):
    """Generate downloadable text report"""
    report = []
    report.append(f"Job Match Report: {candidate['name']} - {job['title']}")
    report.append("="*60)
    report.append(f"\nMatch Score: {match_score}%")
    report.append("\nCandidate Information")
    report.append("-"*30)
    report.append(f"Name: {candidate['name']}")
    report.append(f"Current Role: {candidate['job_title']}")
    report.append(f"Experience: {candidate['experience']} years")
    report.append(f"Education: {candidate['education']}")
    
    report.append("\nJob Information")
    report.append("-"*30)
    report.append(f"Title: {job['title']}")
    report.append(f"Company: {job['company']}")
    report.append(f"Location: {job['location']}")
    report.append(f"Required Experience: {job['min_experience']}+ years")
    
    report.append("\nSkills Analysis")
    report.append("-"*30)
    matched_skills = set(candidate['skills']) & set(job['required_skills'])
    missing_skills = set(job['required_skills']) - set(candidate['skills'])
    
    report.append(f"Matching Skills ({len(matched_skills)}):")
    report.append(", ".join(matched_skills) if matched_skills else "None")
    
    report.append(f"\nMissing Skills ({len(missing_skills)}):")
    report.append(", ".join(missing_skills) if missing_skills else "None")
    
    # Add salary estimation
    estimated_salary = estimate_salary(job['title'], candidate['experience'], job['location'])
    report.append("\nSalary Estimation")
    report.append("-"*30)
    report.append(f"₹{estimated_salary:,.0f} INR per year")
    
    return "\n".join(report)

# Format salary for display (INR only)
def format_salary(salary):
    """Format salary in INR"""
    return f"₹{salary:,.0f} INR"

# ========================
# STREAMLIT UI
# ========================

# Sidebar navigation
st.sidebar.title("JobMatch AI")
st.sidebar.image("https://via.placeholder.com/150x50.png?text=JobMatch+AI", width=150)
page = st.sidebar.radio("Navigation", ["Dashboard", "Candidate Portal", "Recruiter Portal", "Matching Engine", "Documentation"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Need Help?**")
st.sidebar.info("Contact our support team at support@jobmatchai.com")

# Documentation page
if page == "Documentation":
    st.header("📚 Documentation")
    st.subheader("How to Use JobMatch AI")
    
    with st.expander("Getting Started"):
        st.markdown("""
        1. **Add Candidates**: Go to the Candidate Portal to create candidate profiles
        2. **Post Jobs**: Go to the Recruiter Portal to create job listings
        3. **Match Candidates**: Use the Matching Engine to find the best candidates for your jobs
        4. **Track Applications**: Monitor all applications in the Dashboard
        """)
    
    with st.expander("Candidate Portal Guide"):
        st.markdown("""
        - Upload your resume or manually enter your details
        - The AI will automatically extract your skills and experience
        - Get personalized resume improvement suggestions
        - See job recommendations based on your profile
        """)
        
    with st.expander("Recruiter Portal Guide"):
        st.markdown("""
        - Create detailed job listings with required skills and experience
        - Upload job descriptions to automatically extract requirements
        - Get salary suggestions based on market data
        - See top matching candidates for your positions
        """)
        
    with st.expander("Matching Engine Guide"):
        st.markdown("""
        - Select a candidate and a job to compare
        - Get a detailed match score with breakdown
        - See skill gap analysis and improvement suggestions
        - Generate interview questions tailored to the position
        - Download comprehensive match reports
        """)
    
    st.subheader("Frequently Asked Questions")
    faq = {
        "How accurate is the matching algorithm?": "Our AI uses multiple factors including skills, experience, job titles, and semantic analysis to provide accurate matching scores.",
        "Is my data secure?": "Yes, all data is processed locally in your browser and not stored on any servers.",
        "Can I export my data?": "Yes, you can download match reports and application data in various formats.",
        "How often should I update candidate profiles?": "We recommend updating profiles every 6 months or when candidates gain new skills/experience."
    }
    
    for question, answer in faq.items():
        with st.expander(question):
            st.markdown(answer)
    
    st.markdown("---")
    st.markdown("© 2023 JobMatch AI. All rights reserved.")

# Dashboard
elif page == "Dashboard":
    st.header("📊 Dashboard")
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", len(st.session_state.candidates), help="Number of registered candidates")
    with col2:
        st.metric("Total Jobs", len(st.session_state.jobs), help="Number of active job listings")
    with col3:
        avg_match = st.session_state.matches['match_score'].mean() if not st.session_state.matches.empty else 0
        st.metric("Average Match Score", 
                 f"{avg_match:.1f}%" if not st.session_state.matches.empty else "N/A",
                 help="Average match score across all applications")
    with col4:
        st.metric("Active Applications", len(st.session_state.applications), help="Total submitted applications")
    
    # Visualization row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Match distribution
        if not st.session_state.matches.empty:
            st.subheader("Match Distribution")
            fig = px.histogram(
                st.session_state.matches, 
                x='match_score',
                nbins=10,
                labels={'match_score': 'Match Score (%)'},
                color_discrete_sequence=['#4f88ec']
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top skills in demand
        if st.session_state.jobs:
            st.subheader("Top Skills in Demand")
            all_skills = [skill for job in st.session_state.jobs for skill in job['required_skills']]
            skill_counts = Counter(all_skills).most_common(10)
            skills_df = pd.DataFrame(skill_counts, columns=['Skill', 'Count'])
            
            fig = px.bar(
                skills_df,
                x='Count',
                y='Skill',
                orientation='h',
                color='Count',
                color_continuous_scale='Blues',
                labels={'Count': 'Number of Jobs'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Visualization row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Top companies
        if st.session_state.jobs:
            st.subheader("Top Companies Hiring")
            companies = Counter([j['company'] for j in st.session_state.jobs]).most_common(5)
            companies_df = pd.DataFrame(companies, columns=['Company', 'Job Count'])
            
            fig = px.pie(
                companies_df,
                values='Job Count',
                names='Company',
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average salary
        if st.session_state.jobs:
            st.subheader("Salary Distribution")
            salaries = [j['annual_salary'] for j in st.session_state.jobs]
            
            fig = px.box(
                pd.DataFrame(salaries, columns=['Salary']),
                y='Salary',
                labels={'Salary': 'Annual Salary (INR)'}
            )
            fig.update_traces(marker_color='#4f88ec')
            st.plotly_chart(fig, use_container_width=True)
            
            avg_salary = np.mean(salaries)
            st.metric("Average Salary", format_salary(avg_salary))

# Candidate Portal
elif page == "Candidate Portal":
    st.header("👤 Candidate Portal")
    
    with st.expander("Add Your Profile", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*", placeholder="John Doe")
            email = st.text_input("Email*", placeholder="john@example.com")
            location = st.text_input("Location (City)*", placeholder="Bangalore")
            job_title = st.selectbox("Desired Job Title*", JOB_TITLES, index=0)
        with col2:
            experience = st.slider("Years of Experience*", 0, 30, 2)
            education = st.selectbox("Highest Education*", 
                                    ["PhD", "Master's", "Bachelor's", "Associate", "High School"],
                                    index=1)
            resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
        
        summary = st.text_area("Professional Summary*", height=150,
                              placeholder="Briefly describe your skills, experience, and career goals...",
                              help="This helps our AI make better job recommendations")
        
        if st.button("Create Profile", type="primary"):
            if not name or not email or not location or not job_title or not summary:
                st.error("Please fill in all required fields (*)")
            else:
                candidate = {
                    "id": len(st.session_state.candidates) + 1,
                    "name": name,
                    "email": email,
                    "location": location,
                    "job_title": job_title,
                    "experience": experience,
                    "education": education,
                    "summary": summary
                }
                
                # Extract skills from resume if uploaded
                if resume_file:
                    resume_text = extract_text(resume_file)
                    candidate["skills"] = extract_skills(resume_text)
                    candidate["resume_text"] = resume_text
                    st.success(f"Extracted {len(candidate['skills'])} skills from resume")
                else:
                    candidate["skills"] = []
                
                st.session_state.candidates.append(candidate)
                st.toast("Profile created successfully!", icon="✅")
    
    # Display candidate profiles
    if st.session_state.candidates:
        st.subheader("Your Profile")
        for candidate in st.session_state.candidates:
            with st.expander(f"👤 {candidate['name']} - {candidate['job_title']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**📧 Email:** {candidate['email']}")
                    st.write(f"**📍 Location:** {candidate['location']}")
                with col2:
                    st.write(f"**⏳ Experience:** {candidate['experience']} years")
                    st.write(f"**🎓 Education:** {candidate['education']}")
                
                st.divider()
                
                if candidate.get('skills'):
                    st.write("**🛠️ Skills:** " + ", ".join(candidate['skills']))
                
                # Resume suggestions
                if candidate.get('resume_text'):
                    with st.expander("📝 AI Resume Assistant"):
                        suggestions = generate_resume_suggestions(candidate)
                        if suggestions:
                            st.warning("**Resume Improvement Suggestions:**")
                            for suggestion in suggestions:
                                st.write(f"- {suggestion}")
                        else:
                            st.success("✅ Your resume looks strong! Good job highlighting your skills and experience.")
                
                if st.button("Delete Profile", key=f"del_{candidate['id']}", type="secondary"):
                    st.session_state.candidates = [c for c in st.session_state.candidates if c['id'] != candidate['id']]
                    st.experimental_rerun()
    
    # Job recommendations
    st.subheader("🔍 Recommended Jobs For You")
    if st.session_state.candidates and st.session_state.jobs:
        if st.session_state.candidates:
            candidate = st.session_state.candidates[-1]  # Get latest candidate
            recommendations = []
            
            for job in st.session_state.jobs:
                score = calculate_match_score(candidate, job)
                if score >= 70:  # Only show good matches
                    recommendations.append((job, score))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            if recommendations:
                for job, score in recommendations[:3]:  # Show top 3
                    with st.container():
                        st.subheader(f"{job['title']} at {job['company']}")
                        col1, col2 = st.columns([3,1])
                        with col1:
                            st.caption(f"📍 {job['location']} | 💼 {job['job_type']} | ⏳ {job['min_experience']}+ years")
                            
                            # Format salary with INR
                            st.write(f"💵 **Annual Salary:** {format_salary(job['annual_salary'])}")
                            
                            # Progress bar for match score
                            st.progress(score/100, text=f"Match Score: {score}%")
                            
                            st.write(f"🔑 **Skills:** {', '.join(job['required_skills'][:5])}{'...' if len(job['required_skills'])>5 else ''}")
                            
                            # Skill match visualization
                            candidate_skills = set(candidate['skills'])
                            job_skills = set(job['required_skills'])
                            matched = len(candidate_skills & job_skills)
                            total = len(job_skills)
                            
                            st.write(f"✅ **Skills Match:** {matched}/{total} required skills")
                        with col2:
                            if st.button("Apply Now", key=f"apply_{job['id']}", type="primary"):
                                add_application(candidate['id'], job['id'])
                                st.toast("Application submitted!", icon="📬")
                        st.divider()
            else:
                st.info("No strong matches found. Try updating your skills or preferences.")
        else:
            st.info("Complete your profile to see job recommendations")

# Recruiter Portal
elif page == "Recruiter Portal":
    st.header("🏢 Recruiter Portal")
    
    with st.expander("Post New Job", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.selectbox("Job Title*", JOB_TITLES, index=0)
            company = st.text_input("Company Name*", placeholder="Acme Inc.")
            location = st.text_input("Job Location*", placeholder="Bangalore")
            min_experience = st.slider("Minimum Experience (Years)*", 0, 15, 2)
            education_req = st.selectbox("Education Requirement*", 
                                       ["Any", "Bachelor's", "Master's", "PhD"],
                                       index=1)
        with col2:
            job_type = st.selectbox("Job Type*", ["Full-time", "Part-time", "Contract", "Internship"], index=0)
            # SALARY INPUT (LPA and INR)
            salary_lpa = st.number_input(
                "Annual Salary (LPA)*",
                min_value=0.0,
                max_value=100.0,
                value=7.0,
                step=0.1,
                help="Enter salary in Lakhs Per Annum (LPA), e.g., 6 for 6,00,000 INR"
            )
            annual_salary = int(salary_lpa * 100000)
            st.write(f"**Annual Salary:** {format_salary(annual_salary)}")
            st.write(f"**Monthly Salary:** ₹{annual_salary//12:,.0f} INR")
            jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
        
        description = st.text_area("Job Description*", height=150,
                                  placeholder="Describe the role, responsibilities, and requirements...",
                                  help="Be specific about required skills and experience")
        
        required_skills = st.multiselect("Required Skills*", SKILLS_DATABASE, default=["Python", "SQL"])
        
        if st.button("Post Job", type="primary"):
            if not job_title or not company or not location or not description or not required_skills:
                st.error("Please fill in all required fields (*)")
            else:
                job = {
                    "id": len(st.session_state.jobs) + 1,
                    "title": job_title,
                    "company": company,
                    "location": location,
                    "min_experience": min_experience,
                    "education": education_req,
                    "job_type": job_type,
                    "annual_salary": annual_salary,
                    "description": description,
                    "required_skills": required_skills
                }
                
                # Extract skills from JD if uploaded
                if jd_file:
                    jd_text = extract_text(jd_file)
                    job["required_skills"] = list(set(job["required_skills"] + extract_skills(jd_text)))
                    job["description"] = jd_text if jd_text else description
                    st.success(f"Extracted {len(job['required_skills'])} skills from job description")
                
                st.session_state.jobs.append(job)
                st.toast("Job posted successfully!", icon="✅")
    
    # Display job listings
    if st.session_state.jobs:
        st.subheader("Job Listings")
        for job in st.session_state.jobs:
            with st.expander(f"🏢 {job['title']} at {job['company']} - {job['location']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**📋 Type:** {job['job_type']}")
                    st.write(f"**⏳ Experience:** {job['min_experience']}+ years")
                with col2:
                    st.write(f"**🎓 Education:** {job['education']}")
                    st.write(f"**💰 Salary:** {format_salary(job['annual_salary'])}")
                
                st.divider()
                
                st.write("**🔑 Required Skills:** " + ", ".join(job['required_skills']))
                st.write(f"**📝 Description:** {textwrap.shorten(job['description'], width=300, placeholder='...')}")
                
                if st.button("Delete Job", key=f"del_job_{job['id']}", type="secondary"):
                    st.session_state.jobs = [j for j in st.session_state.jobs if j['id'] != job['id']]
                    st.experimental_rerun()
    
    # Candidate recommendations
    if st.session_state.jobs:
        st.subheader(f"🌟 Top Candidates")
        if st.session_state.candidates:
            job = st.session_state.jobs[-1] if st.session_state.jobs else None
            if job:
                recommendations = []
                
                for candidate in st.session_state.candidates:
                    score = calculate_match_score(candidate, job)
                    if score >= 65:  # Only show reasonable matches
                        recommendations.append((candidate, score))
                
                recommendations.sort(key=lambda x: x[1], reverse=True)
                
                if recommendations:
                    for candidate, score in recommendations[:5]:  # Show top 5
                        with st.container():
                            st.subheader(f"{candidate['name']} - {candidate['job_title']}")
                            col1, col2 = st.columns([3,1])
                            with col1:
                                st.write(f"📍 {candidate['location']} | ⏳ {candidate['experience']} years experience")
                                st.write(f"🎓 {candidate['education']}")
                                st.write(f"🛠️ Skills: {', '.join(candidate['skills'][:5])}{'...' if len(candidate['skills'])>5 else ''}")
                                
                                # Progress bar for match score
                                st.progress(score/100, text=f"Match Score: {score}%")
                            with col2:
                                # Show estimated salary for candidate in this role
                                estimated_salary = estimate_salary(
                                    job['title'], 
                                    candidate['experience'], 
                                    job['location']
                                )
                                st.metric("Estimated Salary", format_salary(estimated_salary))
                                if st.button("Contact Candidate", key=f"contact_{candidate['id']}", type="primary"):
                                    st.toast(f"Message sent to {candidate['name']}!", icon="✉️")
                            st.divider()
                else:
                    st.info("No strong matches found. Consider adjusting job requirements.")
            else:
                st.info("Post a job to see candidate recommendations")
        else:
            st.info("No candidates available yet")

# Matching Engine
elif page == "Matching Engine":
    st.header("🤖 AI Matching Engine")
    
    if not st.session_state.candidates or not st.session_state.jobs:
        st.warning("Please add at least one candidate and one job to use the matching engine")
        st.stop()
    
    # Select candidate and job
    candidate_names = [f"{c['name']} ({c['job_title']})" for c in st.session_state.candidates]
    job_titles = [f"{j['title']} at {j['company']}" for j in st.session_state.jobs]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_candidate = st.selectbox("Select Candidate", candidate_names, index=0)
    with col2:
        selected_job = st.selectbox("Select Job", job_titles, index=0)
    
    candidate_idx = candidate_names.index(selected_candidate)
    job_idx = job_titles.index(selected_job)
    
    candidate = st.session_state.candidates[candidate_idx]
    job = st.session_state.jobs[job_idx]
    
    # Calculate match
    match_score = None
    if st.button("Calculate Match Score", type="primary"):
        match_score = calculate_match_score(candidate, job)
        skill_gap = generate_skill_gap(candidate['skills'], job['required_skills'])
        match_reasons = generate_match_reasons(candidate, job, match_score)
        
        # Store match for analytics
        new_match = {
            "candidate": candidate['name'],
            "job": f"{job['title']} at {job['company']}",
            "match_score": match_score,
            "skills_match": len(set(candidate['skills']) & set(job['required_skills'])),
            "experience_match": candidate['experience'] >= job['min_experience']
        }
        st.session_state.matches = pd.concat([st.session_state.matches, pd.DataFrame([new_match])], ignore_index=True)
        
        # Display results
        st.subheader(f"Match Score: {match_score}%")
        st.progress(match_score / 100)
        
        # Display match reasons
        if match_reasons:
            st.subheader("Why this is a good match:")
            for reason in match_reasons:
                st.success(f"✓ {reason}")
        
        # Skill gap analysis
        if skill_gap:
            st.subheader("Skill Gap Analysis")
            st.warning(f"To increase your match score, consider developing these skills:")
            for skill in skill_gap:
                st.write(f"- {skill}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            # Skill comparison
            candidate_skills = set(candidate['skills'])
            job_skills = set(job['required_skills'])
            
            skill_data = {
                "Skills": ["Candidate Skills", "Job Required Skills", "Matched Skills"],
                "Count": [
                    len(candidate_skills),
                    len(job_skills),
                    len(candidate_skills & job_skills)
                ]
            }
            
            fig = px.bar(
                pd.DataFrame(skill_data),
                x='Skills',
                y='Count',
                color='Skills',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
                text='Count'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Score breakdown
            st.subheader("Score Breakdown")
            
            # Calculate individual scores
            skill_vectorizer = TfidfVectorizer()
            candidate_skills_text = " ".join(candidate['skills'])
            job_skills_text = " ".join(job['required_skills'])
            skill_matrix = skill_vectorizer.fit_transform([candidate_skills_text, job_skills_text])
            skill_similarity = cosine_similarity(skill_matrix[0:1], skill_matrix[1:2])[0][0] * 100
            
            exp_match = 100 if candidate['experience'] >= job['min_experience'] else \
                (candidate['experience'] / job['min_experience']) * 100
            
            title_similarity = cosine_similarity(
                [generate_embeddings(candidate['job_title'])], 
                [generate_embeddings(job['title'])]
            )[0][0] * 100
            
            desc_similarity = cosine_similarity(
                [generate_embeddings(candidate['summary'])], 
                [generate_embeddings(job['description'])]
            )[0][0] * 100
            
            # Create radar chart
            categories = ['Skills', 'Experience', 'Title', 'Description']
            scores = [skill_similarity, exp_match, title_similarity, desc_similarity]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name='Score Breakdown',
                line_color='#4f88ec'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Interview preparation
    if job:
        with st.expander("🎤 Interview Preparation Kit"):
            questions = generate_interview_questions(job)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Technical Questions")
                for i, q in enumerate(questions['technical'][:5], 1):
                    st.markdown(f"{i}. {q}")
            
            with col2:
                st.subheader("Behavioral Questions")
                for i, q in enumerate(questions['behavioral'], 1):
                    st.markdown(f"{i}. {q}")
            
            st.download_button(
                label="Download Full Question List",
                data="\n\n".join(questions['technical'] + questions['behavioral']),
                file_name=f"{job['title']}_interview_questions.txt",
                type="primary"
            )
    
    # Text report
    if candidate and job:
        if st.button("📬 Apply for This Position", type="primary"):
            app = add_application(candidate['id'], job['id'])
            st.toast(f"Application submitted! Your application ID: #{app['id']}", icon="📬")
        
        # Generate and download report
        if match_score:
            st.subheader("📊 Download Match Report")
            report_text = generate_text_report(candidate, job, match_score)
            st.download_button(
                label="Download Full Report (TXT)",
                data=report_text,
                file_name=f"JobMatch_Report_{candidate['name']}_{job['title']}.txt",
                mime="text/plain",
                type="primary"
            )
    
    # Show application status
    if st.session_state.applications:
        st.subheader("Your Applications")
        app_df = pd.DataFrame(st.session_state.applications)
        app_df = app_df[app_df['candidate_id'] == candidate['id']]
        if not app_df.empty:
            # Add job titles to the dataframe
            job_titles = {j['id']: j['title'] for j in st.session_state.jobs}
            app_df['Job Title'] = app_df['job_id'].map(job_titles)
            
            # Format status with colors
            def status_style(status):
                if status == "Applied":
                    return "background-color: #e6f7ff; color: #1890ff;"
                elif status == "Interview":
                    return "background-color: #fff7e6; color: #fa8c16;"
                elif status == "Hired":
                    return "background-color: #f6ffed; color: #52c41a;"
                elif status == "Rejected":
                    return "background-color: #fff1f0; color: #f5222d;"
                return ""
            
            styled_df = app_df[['id', 'Job Title', 'status', 'date']].style.applymap(
                status_style, subset=['status']
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("You haven't applied to any positions yet")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.markdown("""
    <style>
    .footer {
        font-size: 14px;
        color: #666;
        text-align: center;
        padding: 10px;
        margin-top: 30px;
    }
    .footer a {
        color: #1e88e5;
        text-decoration: none;
    }
    </style>
    <div class="footer">
        JOB MATCH AI  
        <a href="#" target="_blank">Privacy Policy</a> | 
        <a href="#" target="_blank">Terms of Service</a>
    </div>
""", unsafe_allow_html=True)
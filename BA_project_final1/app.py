import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import re

# Job Category Rules - Define keyword patterns for job categories
JOB_CATEGORY_RULES = {
    'Technology': {
        'keywords': ['software', 'coding', 'programming', 'developer', 'tech', 'technology', 'data science',
                     'machine learning', 'AI', 'artificial intelligence', 'cybersecurity', 'cloud', 'devops'],
        'patterns': [r'\b(software\s*(developer|engineer))', r'\b(data\s*(scientist|analyst))',
                     r'\b(cyber\s*security)', r'\b(cloud\s*(computing|engineer))'],
        'category': 'Technology Roles'
    },
    'Business': {
        'keywords': ['management', 'business', 'marketing', 'finance', 'project manager', 'operations',
                     'strategy', 'sales', 'consulting', 'leadership', 'analytics'],
        'patterns': [r'\b(project\s*manager)', r'\b(business\s*(analyst|development))',
                     r'\b(finance\s*manager)', r'\b(marketing\s*(manager|coordinator))'],
        'category': 'Business & Management Roles'
    },
    'Creative': {
        'keywords': ['design', 'creative', 'arts', 'graphic', 'ux', 'ui', 'content', 'media',
                     'writing', 'video', 'photography', 'branding'],
        'patterns': [r'\b(ux\s*/\s*ui)', r'\b(graphic\s*design)', r'\b(content\s*(creator|manager))',
                     r'\b(digital\s*media)'],
        'category': 'Creative & Arts Roles'
    },
    'Transition': {
        'keywords': ['career change', 'new field', 'transition', 'retraining', 'upskilling',
                     'new opportunity', 'jobless', 'unemployed', 'new skills'],
        'patterns': [r'\b(career\s*(change|transition))', r'\b(new\s*(field|opportunity))',
                     r'\b(upskill(ing)?)'],
        'category': 'Career Transition Roles'
    }
}


# Load models using joblib
@st.cache_resource
def load_models():
    try:
        sentiment_model = joblib.load("sentiment_model.pkl")
        sentiment_vectorizer = joblib.load("sentiment_vectorizer.pkl")
        lda_model = joblib.load("lda_model.pkl")
        lda_vectorizer = joblib.load("topic_vectorizer.pkl")

        recommender_loaded = joblib.load("recommender_model.pkl")
        if isinstance(recommender_loaded, tuple) and len(recommender_loaded) == 4:
            recommender_model, recommender_data, recommender_texts, recommender_encoders = recommender_loaded
        else:
            recommender_model = recommender_loaded
            recommender_data = pd.DataFrame(columns=["Age", "Gender", "Education", "Field", "Employed"])
            recommender_texts = pd.Series(["Sample response 1", "Sample response 2", "Sample response 3"])
            recommender_encoders = {
                "Age": LabelEncoder(),
                "Gender": LabelEncoder(),
                "Education": LabelEncoder(),
                "Field": LabelEncoder(),
                "Employed": LabelEncoder()
            }
            for col, encoder in recommender_encoders.items():
                if col == "Age":
                    encoder.fit(["18–24", "25–30", "31–35", "36+"])
                elif col == "Gender":
                    encoder.fit(["Male", "Female", "Other"])
                elif col == "Education":
                    encoder.fit(["Bachelor's Degree", "Postgraduate Degree", "Diploma", "High School"])
                elif col == "Field":
                    encoder.fit(["Science & Technology", "Business & Management", "Arts", "Other"])
                elif col == "Employed":
                    encoder.fit(["Yes", "No"])

        segmentation_loaded = joblib.load("segmentation_model.pkl")
        if isinstance(segmentation_loaded, tuple) and len(segmentation_loaded) == 2:
            segmentation_model, segmentation_encoders = segmentation_loaded
        else:
            segmentation_model = segmentation_loaded
            segmentation_encoders = {
                "Age": LabelEncoder(),
                "Gender": LabelEncoder(),
                "Education": LabelEncoder(),
                "Field": LabelEncoder(),
                "Employed": LabelEncoder()
            }
            for col, encoder in segmentation_encoders.items():
                if col == "Age":
                    encoder.fit(["18–24", "25–30", "31–35", "36+"])
                elif col == "Gender":
                    encoder.fit(["Male", "Female", "Other"])
                elif col == "Education":
                    encoder.fit(["Bachelor's Degree", "Postgraduate Degree", "Diploma/certificate", "Others"])
                elif col == "Field":
                    encoder.fit(["Science & Technology", "Business & Management", "Arts", "Other"])
                elif col == "Employed":
                    encoder.fit(["Yes", "No"])

        return (sentiment_model, sentiment_vectorizer, lda_model, lda_vectorizer,
                recommender_model, recommender_data, recommender_texts, recommender_encoders,
                segmentation_model, segmentation_encoders)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please check that all model files exist and are properly saved.")
        return None


def analyze_sentiment_enhanced(text, sentiment_model, sentiment_vectorizer):
    if not text or text.strip() == "":
        return None, None, None

    try:
        vec = sentiment_vectorizer.transform([text])
        sentiment_pred = sentiment_model.predict(vec)[0]
        try:
            sentiment_proba = sentiment_model.predict_proba(vec)[0]
            confidence = max(sentiment_proba)
        except:
            confidence = 0.75

        sentiment_mapping = {
            0: "Critical",
            1: "Neutral",
            2: "Constructive"
        }
        sentiment_label = sentiment_mapping.get(sentiment_pred, f"Unknown ({sentiment_pred})")

        if sentiment_pred == 0:
            polarity = -0.5 - (confidence - 0.5) * 0.5
        elif sentiment_pred == 1:
            polarity = 0.0
        elif sentiment_pred == 2:
            polarity = 0.5 + (confidence - 0.5) * 0.5
        else:
            polarity = 0.0

        return sentiment_label, confidence, polarity

    except Exception as e:
        st.warning(f"Model prediction failed, using fallback method: {str(e)}")
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                          'happy', 'satisfied', 'love', 'best', 'perfect', 'awesome',
                          'constructive', 'helpful', 'beneficial', 'valuable', 'useful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
                          'disappointed', 'frustrated', 'difficult', 'challenging', 'problems',
                          'critical', 'concerning', 'disappointing', 'inadequate', 'poor']
        neutral_words = ['okay', 'fine', 'average', 'normal', 'standard', 'typical']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_words if word in text_lower)

        total_words = len(text.split())
        sentiment_words = positive_count + negative_count + neutral_count

        if sentiment_words == 0:
            return "Neutral", 0.5, 0.0

        confidence = min(0.9, 0.5 + (sentiment_words / total_words) * 0.4)

        if positive_count > negative_count and positive_count > neutral_count:
            polarity = 0.3 + (positive_count / (positive_count + negative_count + neutral_count)) * 0.7
            return "Constructive", confidence, polarity
        elif negative_count > positive_count and negative_count > neutral_count:
            polarity = -0.3 - (negative_count / (positive_count + negative_count + neutral_count)) * 0.7
            return "Critical", confidence, polarity
        else:
            return "Neutral", confidence, 0.0


def rule_based_job_classification(text):
    """Rule-based classification for job categories based on text input"""
    if not text or text.strip() == "":
        return "Uncategorized", 0.0

    text_lower = text.lower()
    category_scores = {}

    for category, rules in JOB_CATEGORY_RULES.items():
        score = 0
        # Keyword matching
        for keyword in rules['keywords']:
            if keyword in text_lower:
                score += 0.3  # Base score for keyword match

        # Pattern matching
        for pattern in rules['patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.5  # Higher score for pattern match

        category_scores[category] = score

    # Find the category with the highest score
    if not category_scores or max(category_scores.values()) == 0:
        return "Uncategorized", 0.0

    best_category = max(category_scores, key=category_scores.get)
    confidence = min(1.0, category_scores[best_category])

    return JOB_CATEGORY_RULES[best_category]['category'], confidence


def hybrid_job_classification(text, sentiment_model, sentiment_vectorizer):
    """Hybrid approach combining rule-based and ML-based classification"""
    # Get rule-based classification
    rule_category, rule_confidence = rule_based_job_classification(text)

    # Get sentiment analysis
    sentiment_label, sentiment_confidence, _ = analyze_sentiment_enhanced(text, sentiment_model, sentiment_vectorizer)

    # Combine results with weighted scoring
    final_category = rule_category
    final_confidence = rule_confidence

    # Adjust based on sentiment
    if sentiment_label == "Critical" and rule_category != "Career Transition Roles":
        # If sentiment is critical, increase likelihood of transition category
        transition_score = JOB_CATEGORY_RULES['Transition']['keywords'].count(text.lower()) * 0.3
        if transition_score > rule_confidence:
            final_category = "Career Transition Roles"
            final_confidence = transition_score

    return final_category, final_confidence


def create_sentiment_visualization(sentiment_label, confidence, polarity):
    color_mapping = {
        'Constructive': '#4CAF50',
        'Positive': '#4CAF50',
        'Critical': '#F44336',
        'Negative': '#F44336',
        'Neutral': '#FF9800'
    }
    color = color_mapping.get(sentiment_label, '#9E9E9E')

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=polarity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Sentiment: {sentiment_label}"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [-1, -0.33], 'color': "lightcoral"},
                {'range': [-0.33, 0.33], 'color': "lightyellow"},
                {'range': [0.33, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': polarity
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    return fig_gauge


# Enhanced cluster characteristics (unchanged)
CLUSTER_FEATURES = {
    0: {
        "name": "Tech-Savvy Graduates",
        "characteristics": [
            "Primarily STEM backgrounds",
            "High employment rate in tech sector",
            "Positive sentiment towards career prospects",
            "Focus on innovation and technology topics"
        ],
        "dominant_topics": ["Technology", "Innovation", "Career Growth"],
        "employment_outlook": "Excellent",
        "sentiment_trend": "Positive",
        "recommended_jobs": [
            {"title": "Software Developer"},
            {"title": "Data Scientist"},
            {"title": "Cybersecurity Analyst"}
        ],
        "skill_recommendations": [
            {"skill": "Cloud Computing (AWS/Azure)", "priority": "High", "reason": "Essential for modern tech roles"},
            {"skill": "Machine Learning/AI", "priority": "High", "reason": "Rapidly growing field with high demand"},
            {"skill": "DevOps & CI/CD", "priority": "Medium", "reason": "Critical for software development lifecycle"}
        ]
    },
    1: {
        "name": "Business & Management Professionals",
        "characteristics": [
            "Business and management education",
            "Mixed employment status",
            "Neutral to positive career sentiment",
            "Focus on leadership and business strategy"
        ],
        "dominant_topics": ["Management", "Leadership", "Business Strategy"],
        "employment_outlook": "Good",
        "sentiment_trend": "Neutral-Positive",
        "recommended_jobs": [
            {"title": "Project Manager"},
            {"title": "Business Analyst"},
            {"title": "Digital Marketing Manager"}
        ],
        "skill_recommendations": [
            {"skill": "Data Analytics & Visualization", "priority": "High",
             "reason": "Data-driven decision making is crucial"},
            {"skill": "Agile/Scrum Methodology", "priority": "High", "reason": "Standard in modern project management"},
            {"skill": "Digital Marketing & SEO", "priority": "Medium",
             "reason": "Essential for business growth in digital age"}
        ]
    },
    2: {
        "name": "Creative & Arts Graduates",
        "characteristics": [
            "Arts and humanities background",
            "Variable employment patterns",
            "Mixed sentiment about job market",
            "Focus on creativity and self-expression"
        ],
        "dominant_topics": ["Creativity", "Expression", "Culture"],
        "employment_outlook": "Moderate",
        "sentiment_trend": "Mixed",
        "recommended_jobs": [
            {"title": "UX/UI Designer"},
            {"title": "Content Creator/Manager"},
            {"title": "Digital Media Specialist"}
        ],
        "skill_recommendations": [
            {"skill": "UI/UX Design Tools (Figma, Adobe XD)", "priority": "High",
             "reason": "High demand in digital product development"},
            {"skill": "Video Production & Editing", "priority": "High",
             "reason": "Growing demand in content marketing"},
            {"skill": "Social Media Analytics", "priority": "Medium",
             "reason": "Essential for measuring creative impact"}
        ]
    },
    3: {
        "name": "Career Transitioners",
        "characteristics": [
            "Diverse educational backgrounds",
            "Currently seeking new opportunities",
            "Cautious optimism about future",
            "Focus on skill development and adaptation"
        ],
        "dominant_topics": ["Career Change", "Skill Development", "Adaptation"],
        "employment_outlook": "Improving",
        "sentiment_trend": "Cautiously Optimistic",
        "recommended_jobs": [
            {"title": "Business Development Representative"},
            {"title": "Training & Development Specialist"},
            {"title": "Operations Manager"}
        ],
        "skill_recommendations": [
            {"skill": "Professional Networking & LinkedIn", "priority": "High",
             "reason": "Critical for career transition success"},
            {"skill": "Certification in New Field", "priority": "High",
             "reason": "Validates expertise in target industry"},
            {"skill": "Cross-functional Communication", "priority": "Medium",
             "reason": "Essential for adapting to new work environments"}
        ]
    }
}

# Job market insights (unchanged)
JOB_FIELD_MAPPING = {
    ("Science & Technology", "Bachelor's Degree"): [
        {"title": "Junior Software Engineer", "growth": "22%", "avg_salary": "$72,000"},
        {"title": "Quality Assurance Analyst", "growth": "10%", "avg_salary": "$55,570"},
        {"title": "Technical Support Specialist", "growth": "8%", "avg_salary": "$54,760"}
    ],
    ("Science & Technology", "Postgraduate Degree"): [
        {"title": "Senior Data Scientist", "growth": "35%", "avg_salary": "$126,830"},
        {"title": "Research Scientist", "growth": "8%", "avg_salary": "$89,220"},
        {"title": "Solutions Architect", "growth": "25%", "avg_salary": "$134,730"}
    ],
    ("Business & Management", "Bachelor's Degree"): [
        {"title": "Sales Representative", "growth": "4%", "avg_salary": "$56,152"},
        {"title": "Human Resources Specialist", "growth": "10%", "avg_salary": "$63,490"},
        {"title": "Marketing Coordinator", "growth": "18%", "avg_salary": "$46,680"}
    ],
    ("Business & Management", "Postgraduate Degree"): [
        {"title": "Management Consultant", "growth": "14%", "avg_salary": "$87,660"},
        {"title": "Finance Manager", "growth": "17%", "avg_salary": "$134,180"},
        {"title": "Operations Director", "growth": "9%", "avg_salary": "$125,660"}
    ],
    ("Arts", "Bachelor's Degree"): [
        {"title": "Graphic Designer", "growth": "3%", "avg_salary": "$47,640"},
        {"title": "Social Media Coordinator", "growth": "17%", "avg_salary": "$41,000"},
        {"title": "Content Writer", "growth": "12%", "avg_salary": "$48,250"}
    ],
    ("Arts", "Postgraduate Degree"): [
        {"title": "Creative Director", "growth": "11%", "avg_salary": "$97,270"},
        {"title": "Art Director", "growth": "11%", "avg_salary": "$94,220"},
        {"title": "Museum Curator", "growth": "13%", "avg_salary": "$54,560"}
    ],
    ("Science & Technology", "Diploma"): [
        {"title": "Web Developer (Entry-Level)", "growth": "22%", "avg_salary": "$72,000"},
        {"title": "Electronics Technician", "growth": "10%", "avg_salary": "$55,570"},
        {"title": "Lab Technician", "growth": "8%", "avg_salary": "$54,760"}
    ],
    ("Business & Management", "Diploma"): [
        {"title": "Marketing Assistant", "growth": "4%", "avg_salary": "$56,152"},
        {"title": "Business Development Executive", "growth": "10%", "avg_salary": "$63,490"},
        {"title": "Project Coordinator", "growth": "18%", "avg_salary": "$46,680"}
    ],
    ("Arts", "Diploma"): [
        {"title": "Administrative Assistant", "growth": "11%", "avg_salary": "$97,270"},
        {"title": "Public Relations Assistant", "growth": "11%", "avg_salary": "$94,220"},
        {"title": "Photographer / Videographer", "growth": "13%", "avg_salary": "$54,560"}
    ],
    ("Science & Technology", "Other"): [
        {"title": "Freelance Content Moderator", "growth": "22%", "avg_salary": "$72,000"},
        {"title": "Environmental Scientist", "growth": "10%", "avg_salary": "$55,570"},
        {"title": "Software Developer", "growth": "8%", "avg_salary": "$54,760"}
    ],
    ("Business & Management", "Other"): [
        {"title": "Entrepreneur", "growth": "4%", "avg_salary": "$56,152"},
        {"title": "Accountant", "growth": "10%", "avg_salary": "$63,490"},
        {"title": "Marketing Coordinator", "growth": "18%", "avg_salary": "$46,680"}
    ],
    ("Arts", "Other"): [
        {"title": "Graphic Designer", "growth": "11%", "avg_salary": "$97,270"},
        {"title": "Concept Artist", "growth": "11%", "avg_salary": "$94,220"},
        {"title": "Photographer", "growth": "13%", "avg_salary": "$54,560"}
    ]

}


def get_personalized_recommendations(age, education, field, employment_status, sentiment, cluster_id, job_category):
    """Generate personalized job and skill recommendations with job category consideration"""
    cluster_info = CLUSTER_FEATURES.get(cluster_id, CLUSTER_FEATURES[0])
    base_jobs = cluster_info["recommended_jobs"]
    base_skills = cluster_info["skill_recommendations"]
    field_jobs = JOB_FIELD_MAPPING.get((field, education), [])

    all_jobs = base_jobs + field_jobs

    # Adjust recommendations based on job category
    if job_category != "Uncategorized":
        category_specific_jobs = [
            job for job in all_jobs if any(keyword in job['title'].lower()
                                           for keyword in JOB_CATEGORY_RULES[job_category.split()[0]]['keywords'])
        ]
        if category_specific_jobs:
            all_jobs = category_specific_jobs + all_jobs[:2]  # Prioritize category-specific jobs

    if employment_status == "No":
        entry_level_jobs = [
            {"title": "Internship Programs", "growth": "15%", "avg_salary": "$35,000"},
            {"title": "Graduate Trainee", "growth": "12%", "avg_salary": "$42,000"},
            {"title": "Junior Analyst", "growth": "18%", "avg_salary": "$48,500"}
        ]
        all_jobs.extend(entry_level_jobs)

    enhanced_skills = base_skills.copy()
    if sentiment == "Critical" or employment_status == "No":
        enhanced_skills.append({
            "skill": "Interview & Resume Building",
            "priority": "High",
            "reason": "Essential for job search success"
        })

    if age in ["31–35", "36+"]:
        enhanced_skills.append({
            "skill": "Leadership & Management",
            "priority": "Medium",
            "reason": "Valuable for senior-level positions"
        })

    # Add category-specific skills
    if job_category != "Uncategorized":
        category_key = job_category.split()[0]
        if category_key == 'Technology':
            enhanced_skills.append(
                {"skill": "Python Programming", "priority": "High", "reason": "Essential for tech roles"})
        elif category_key == 'Business':
            enhanced_skills.append(
                {"skill": "Business Strategy", "priority": "High", "reason": "Key for management roles"})
        elif category_key == 'Creative':
            enhanced_skills.append(
                {"skill": "Adobe Creative Suite", "priority": "High", "reason": "Standard for creative roles"})

    unique_jobs = []
    seen_titles = set()
    for job in all_jobs:
        if job["title"] not in seen_titles:
            unique_jobs.append(job)
            seen_titles.add(job["title"])

    return unique_jobs[:3], enhanced_skills[:4]


# App Configuration
st.set_page_config(page_title="Graduate Intelligence Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (unchanged)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .cluster-info {
        background: #49094f;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #bbdefb;
        margin: 1rem 0;
    }
    .job-card {
        background: #0d6114;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
    .skill-card {
        background: #0d6114;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
    .sentiment-metrics {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .priority-high { color: #d32f2f; font-weight: bold; }
    .priority-medium { color: #f57c00; font-weight: bold; }
    .priority-low { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Main Header (unchanged)
st.markdown("""
<div class="main-header">
    <h1>Workforce Readiness Intelligence Dashboard</h1>
    <p>Comprehensive Analysis: Sentiment • Topics • Recommendations • Career Guidance • Skill Development</p>
</div>
""", unsafe_allow_html=True)

# Load models
models = load_models()
if models is None:
    st.stop()

(sentiment_model, sentiment_vectorizer, lda_model, lda_vectorizer,
 recommender_model, recommender_data, recommender_texts, recommender_encoders,
 segmentation_model, segmentation_encoders) = models

# Create columns for input and results
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📝 Your Information")
    with st.form("graduate_form"):
        st.markdown("#### Personal Details")
        age = st.selectbox("Age Range", ["18–24", "25–30", "31–35", "36+"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        st.markdown("#### Education & Career")
        education = st.selectbox("Education Level",
                                 ["Bachelor's Degree", "Postgraduate Degree", "Diploma", "Other"])
        field_of_study = st.selectbox("Field of Study",
                                      ["Science & Technology", "Business & Management", "Arts", "Other"])
        currently_employed = st.selectbox("Employment Status", ["Yes", "No"])
        st.markdown("#### Your Thoughts & Goals")
        feedback_text = st.text_area("Share your feedback about your educational experience:",
                                     placeholder="Tell us about your overall experience, challenges, and thoughts...")
        career_description = st.text_area("Describe your career goals and background:",
                                          placeholder="Describe your field, career goals, and aspirations...")
        topic_text = st.text_area("Share any additional thoughts or responses:",
                                  placeholder="Any other thoughts, concerns, or topics you'd like to discuss...")
        submitted = st.form_submit_button("🚀 Analyze My Profile", use_container_width=True)

with col2:
    st.markdown("### 📊 Analysis Results")
    if submitted:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [" 📈 Overview", " 💭 Sentiment", " 🧠 Topics", " 🎯 Segment", " 💼 Career Guidance"])

        with tab1:
            st.markdown("#### 🔍 Comprehensive Analysis Overview")
            results = {}

            # Sentiment Analysis
            if feedback_text:
                sentiment_label, confidence, polarity = analyze_sentiment_enhanced(
                    feedback_text, sentiment_model, sentiment_vectorizer)
                if sentiment_label:
                    results['sentiment'] = sentiment_label
                    results['sentiment_confidence'] = confidence
                    results['sentiment_polarity'] = polarity
                else:
                    results['sentiment'] = "Unable to analyze"
                    results['sentiment_confidence'] = 0.0
                    results['sentiment_polarity'] = 0.0

            # Job Category Classification
            combined_text = (feedback_text + " " + career_description + " " + topic_text).strip()
            if combined_text:
                job_category, job_confidence = hybrid_job_classification(combined_text, sentiment_model,
                                                                         sentiment_vectorizer)
                results['job_category'] = job_category
                results['job_confidence'] = job_confidence

            # Topic Analysis
            if topic_text:
                try:
                    vec = lda_vectorizer.transform([topic_text])
                    topics = lda_model.transform(vec)
                    top_topic = topics.argmax()
                    terms = lda_model.components_[top_topic]
                    top_words = [lda_vectorizer.get_feature_names_out()[i] for i in terms.argsort()[-5:][::-1]]
                    results['topic'] = f"Topic {top_topic}"
                    results['keywords'] = top_words
                except Exception as e:
                    results['topic'] = "Unable to identify"
                    results['keywords'] = []

            # Segmentation
            try:
                expected_seg_columns = list(segmentation_encoders.keys())
                user_values = [age, gender, education, field_of_study, currently_employed]
                if len(expected_seg_columns) == len(user_values):
                    user_input = pd.DataFrame([user_values], columns=expected_seg_columns)
                else:
                    user_input = pd.DataFrame([[age, gender, education, field_of_study, currently_employed]],
                                              columns=["Age", "Gender", "Education", "Field", "Employed"])
                for col in user_input.columns:
                    if col in segmentation_encoders:
                        le = segmentation_encoders[col]
                        if user_input[col][0] not in le.classes_:
                            le.classes_ = np.append(le.classes_, user_input[col][0])
                        user_input[col] = le.transform(user_input[col])
                cluster = segmentation_model.predict(user_input)[0]
                results['cluster'] = cluster
            except Exception as e:
                if field_of_study == "Science & Technology" and age in ["18–24", "25–30"]:
                    results['cluster'] = 0
                elif field_of_study == "Business & Management":
                    results['cluster'] = 1
                elif field_of_study == "Arts":
                    results['cluster'] = 2
                elif age in ["31–35", "36+"] or currently_employed == "No":
                    results['cluster'] = 3
                else:
                    user_hash = hash(f"{age}-{gender}-{education}-{field_of_study}-{currently_employed}")
                    results['cluster'] = abs(user_hash) % 4

            # Display overview metrics
            col_metrics = st.columns(4)
            with col_metrics[0]:
                st.metric("Sentiment", results.get('sentiment', 'N/A'))
            with col_metrics[1]:
                employment_display = "Yes" if currently_employed == "Yes" else "No"
                st.metric("Currently Employed", employment_display)
            with col_metrics[2]:
                st.metric("Dominant Topic", results.get('topic', 'N/A'))
            with col_metrics[3]:
                st.metric("Job Category", results.get('job_category', 'N/A'))

            if 'cluster' in results:
                cluster_id = results['cluster']
                if cluster_id in CLUSTER_FEATURES:
                    cluster_info = CLUSTER_FEATURES[cluster_id]
                    st.markdown(f"""
                    <div class="cluster-info">
                        <h3>🎯 Your Segment: {cluster_info['name']}</h3>
                        <p><strong>Employment Outlook:</strong> {cluster_info['employment_outlook']}</p>
                        <p><strong>Sentiment Trend:</strong> {cluster_info['sentiment_trend']}</p>
                        <p><strong>Job Category:</strong> {results.get('job_category', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            st.markdown("#### 💭 Sentiment Analysis")
            if feedback_text and 'sentiment' in results:
                sentiment = results['sentiment']
                fig = create_sentiment_visualization(sentiment, results['sentiment_confidence'],
                                                     results['sentiment_polarity'])
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"**Detected Sentiment:** {sentiment}")
                interpretations = {
                    'Positive': "Your feedback shows optimism and satisfaction with your educational journey.",
                    'Negative': "Your feedback indicates some concerns or challenges in your experience.",
                    'Neutral': "Your feedback shows a balanced perspective on your educational experience."
                }
                st.info(interpretations.get(sentiment, "Unable to interpret sentiment."))
            else:
                st.warning("Please provide feedback text for sentiment analysis.")

        with tab3:
            st.markdown("#### 🧠 Topic Analysis")
            if topic_text and 'topic' in results:
                st.success(f"**Dominant Topic:** {results['topic']}")
                if results.get('keywords'):
                    keywords_df = pd.DataFrame({
                        'keyword': results['keywords'],
                        'importance': [5, 4, 3, 2, 1]
                    })
                    fig = px.bar(keywords_df, x='importance', y='keyword',
                                 orientation='h', title="Top Keywords from Your Response")
                    st.plotly_chart(fig, use_container_width=True)
                    st.code(", ".join(results['keywords']))
            else:
                st.warning("Please provide text for topic analysis.")

        with tab4:
            st.markdown("#### 🎯 Detailed Segmentation Analysis")
            if 'cluster' in results:
                cluster_id = results['cluster']
                try:
                    user_input_rec = pd.DataFrame([[age, gender, education, field_of_study, currently_employed]],
                                                  columns=recommender_data.columns)
                    for col in user_input_rec.columns:
                        le = recommender_encoders[col]
                        if user_input_rec[col][0] not in le.classes_:
                            le.classes_ = np.append(le.classes_, user_input_rec[col][0])
                        user_input_rec[col] = le.transform(user_input_rec[col])
                    distances, indices = recommender_model.kneighbors(user_input_rec)
                    st.markdown("**Similar Profiles in Your Segment:**")
                    for i, idx in enumerate(indices[0][:3]):
                        st.write(f"{i + 1}. {recommender_texts.iloc[idx]}")
                except Exception as e:
                    st.warning("Unable to find similar profiles.")
                if cluster_id in CLUSTER_FEATURES:
                    cluster_info = CLUSTER_FEATURES[cluster_id]
                    st.markdown(f"### {cluster_info['name']} - Detailed Profile")
                    col_char, col_topics = st.columns(2)
                    with col_char:
                        st.markdown("**Key Characteristics:**")
                        for char in cluster_info['characteristics']:
                            st.write(f"• {char}")
                    with col_topics:
                        st.markdown("**Common Topics of Interest:**")
                        for topic in cluster_info['dominant_topics']:
                            st.write(f"• {topic}")
                    metrics = ['Employment Outlook', 'Sentiment', 'Career Satisfaction', 'Skill Match',
                               'Network Strength']
                    cluster_scores = {
                        0: [0.9, 0.8, 0.85, 0.9, 0.7],
                        1: [0.8, 0.7, 0.8, 0.75, 0.85],
                        2: [0.6, 0.6, 0.7, 0.8, 0.9],
                        3: [0.7, 0.65, 0.6, 0.7, 0.8]
                    }
                    scores = cluster_scores.get(cluster_id, [0.7, 0.7, 0.7, 0.7, 0.7])
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=metrics,
                        fill='toself',
                        name=cluster_info['name']
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        title=f"{cluster_info['name']} - Profile Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.markdown("#### 💼 Personalized Career Guidance")
            if 'cluster' in results and 'job_category' in results:
                cluster_id = results['cluster']
                user_sentiment = results.get('sentiment', 'Neutral')
                job_category = results.get('job_category', 'Uncategorized')
                recommended_jobs, recommended_skills = get_personalized_recommendations(
                    age, education, field_of_study, currently_employed, user_sentiment, cluster_id, job_category
                )
                st.markdown("### 🎯 Recommended Job Fields for You")
                st.markdown(f"*Based on your profile, education, and detected job category: {job_category}*")
                for i, job in enumerate(recommended_jobs, 1):
                    st.markdown(f"""
                    <div class="job-card">
                        <h4>{i}. {job['title']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("### 🚀 Skills You Should Develop")
                st.markdown("*Priority skills to enhance your career prospects*")
                for i, skill in enumerate(recommended_skills, 1):
                    priority_class = f"priority-{skill['priority'].lower()}"
                    st.markdown(f"""
                    <div class="skill-card">
                        <h4>{i}. {skill['skill']} <span class="{priority_class}">({skill['priority']} Priority)</span></h4>
                        <p><strong>Why important:</strong> {skill['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("### 📊 Career Insights & Action Plan")
                col_insights = st.columns(2)
                with col_insights[0]:
                    st.markdown("#### 💡 Immediate Actions")
                    if currently_employed == "No":
                        st.write("🔹 Focus on networking and job applications")
                        st.write("🔹 Consider internships or contract work")
                        st.write("🔹 Update LinkedIn profile and resume")
                    else:
                        st.write("🔹 Identify skill gaps for promotion")
                        st.write("🔹 Seek mentorship opportunities")
                        st.write("🔹 Build professional network in target roles")
                    if user_sentiment == "Critical":
                        st.write("🔹 Consider career counseling")
                        st.write("🔹 Explore alternative career paths")
                with col_insights[1]:
                    st.markdown("#### 📈 Long-term Strategy")
                    if age in ["18–24", "25–30"]:
                        st.write("🔹 Build specialized expertise")
                        st.write("🔹 Consider advanced certifications")
                        st.write("🔹 Gain diverse project experience")
                    else:
                        st.write("🔹 Develop leadership capabilities")
                        st.write("🔹 Consider management roles")
                        st.write("🔹 Leverage experience for consulting")
                    if education == "Bachelor's Degree":
                        st.write("🔹 Consider pursuing advanced degree")
                st.markdown("#### 🌐 Industry Trends Relevant to You")
                industry_trends = {
                    "Science & Technology": [
                        "AI and Machine Learning adoption accelerating",
                        "Cloud computing skills in high demand",
                        "Cybersecurity concerns driving job growth"
                    ],
                    "Business & Management": [
                        "Digital transformation changing business roles",
                        "Data-driven decision making becoming standard",
                        "Remote work creating new management challenges"
                    ],
                    "Arts": [
                        "Digital content creation opportunities growing",
                        "UX/UI design in high demand",
                        "Personal branding becoming crucial"
                    ],
                    "Other": [
                        "Cross-functional skills increasingly valuable",
                        "Continuous learning essential for career growth",
                        "Networking more important than ever"
                    ]
                }
                trends = industry_trends.get(field_of_study, industry_trends["Other"])
                for trend in trends:
                    st.write(f"📈 {trend}")
                st.markdown("#### 📚 Recommended Learning Platforms")
                learning_resources = {
                    "Technical Skills": ["Coursera", "edX", "Udacity", "Pluralsight"],
                    "Business Skills": ["LinkedIn Learning", "Harvard Business Review", "Coursera Business"],
                    "Creative Skills": ["Skillshare", "Adobe Creative Cloud Tutorials", "Domestika"],
                    "General Development": ["Khan Academy", "TED-Ed", "YouTube Learning Channels"]
                }
                col_resources = st.columns(2)
                for i, (category, resources) in enumerate(learning_resources.items()):
                    with col_resources[i % 2]:
                        st.write(f"**{category}:**")
                        for resource in resources:
                            st.write(f"• {resource}")
            else:
                st.warning("Please complete the analysis in other tabs first to see career guidance.")

    else:
        st.info("👆 Fill in your information and click 'Analyze My Profile' to see comprehensive results!")
        st.markdown("### 🎯 Sample Segment Profiles & Career Paths")
        for cluster_id, info in CLUSTER_FEATURES.items():
            with st.expander(f"Cluster {cluster_id}: {info['name']}"):
                col_info, col_career = st.columns(2)
                with col_info:
                    st.write(f"**Employment Outlook:** {info['employment_outlook']}")
                    st.write(f"**Sentiment Trend:** {info['sentiment_trend']}")
                    st.write("**Characteristics:**")
                    for char in info['characteristics']:
                        st.write(f"• {char}")
                with col_career:
                    st.write("**Top Recommended Jobs:**")
                    for job in info['recommended_jobs'][:2]:
                        st.write(f"• {job['title']} ")
                    st.write("**Key Skills to Develop:**")
                    for skill in info['skill_recommendations'][:2]:
                        st.write(f"• {skill['skill']} ({skill['priority']} Priority)")

# Footer (unchanged)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <h4>🎓 Workforce Readiness Intelligence Dashboard</h4>
    <p><strong>Master of Data Analytics - Business Analytics Seminar</strong> | Group 05 </p>
</div>
""", unsafe_allow_html=True)
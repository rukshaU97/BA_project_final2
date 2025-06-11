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

# Text Category Rules - Define keyword patterns for general text categories
TEXT_CATEGORY_RULES = {
    'Positive': {
        'keywords': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy',
                     'satisfied', 'love', 'best', 'perfect', 'awesome', 'helpful', 'beneficial'],
        'patterns': [r'\b(positive\s*experience)', r'\b(highly\s*satisfied)', r'\b(great\s*learning)'],
        'category': 'Positive Feedback'
    },
    'Critical': {
        'keywords': ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointed', 'frustrated',
                     'difficult', 'challenging', 'problems', 'critical', 'concerning', 'inadequate'],
        'patterns': [r'\b(poor\s*experience)', r'\b(dissatisfied)', r'\b(needs\s*improvement)'],
        'category': 'Critical Feedback'
    },
    'Skill': {
        'keywords': ['skill', 'learn', 'training', 'development', 'education', 'upskill', 'course',
                     'certification', 'study', 'improve', 'knowledge'],
        'patterns': [r'\b(skill\s*development)', r'\b(learn(ing)?\s*new)', r'\b(seek\s*training)'],
        'category': 'Skill Development Focus'
    },
    'Inquiry': {
        'keywords': ['question', 'inquiry', 'wonder', 'curious', 'information', 'details', 'ask',
                     'explore', 'interested', 'seek'],
        'patterns': [r'\b(have\s*a\s*question)', r'\b(looking\s*for\s*information)', r'\b(curious\s*about)'],
        'category': 'General Inquiry'
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


def rule_based_text_classification(text):
    """Rule-based classification for general text categories"""
    if not text or text.strip() == "":
        return "Uncategorized", 0.0

    text_lower = text.lower()
    category_scores = {}

    for category, rules in TEXT_CATEGORY_RULES.items():
        score = 0
        for keyword in rules['keywords']:
            if keyword in text_lower:
                score += 0.3
        for pattern in rules['patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.5
        category_scores[category] = score

    if not category_scores or max(category_scores.values()) == 0:
        return "Uncategorized", 0.0

    best_category = max(category_scores, key=category_scores.get)
    confidence = min(1.0, category_scores[best_category])

    return TEXT_CATEGORY_RULES[best_category]['category'], confidence


def hybrid_text_classification(text, sentiment_model, sentiment_vectorizer):
    """Hybrid approach combining rule-based and ML-based classification"""
    rule_category, rule_confidence = rule_based_text_classification(text)
    sentiment_label, sentiment_confidence, _ = analyze_sentiment_enhanced(text, sentiment_model, sentiment_vectorizer)

    final_category = rule_category
    final_confidence = rule_confidence

    # Adjust based on sentiment
    if sentiment_label == "Critical" and rule_category != "Critical Feedback":
        critical_score = sum(
            1 for keyword in TEXT_CATEGORY_RULES['Critical']['keywords'] if keyword in text.lower()) * 0.3
        if critical_score > rule_confidence:
            final_category = "Critical Feedback"
            final_confidence = critical_score
    elif sentiment_label == "Constructive" and rule_category != "Positive Feedback":
        positive_score = sum(
            1 for keyword in TEXT_CATEGORY_RULES['Positive']['keywords'] if keyword in text.lower()) * 0.3
        if positive_score > rule_confidence:
            final_category = "Positive Feedback"
            final_confidence = positive_score

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


# Updated cluster characteristics without job roles
CLUSTER_FEATURES = {
    0: {
        "name": "Tech-Savvy Graduates",
        "characteristics": [
            "Primarily STEM backgrounds",
            "High engagement in tech-related topics",
            "Positive sentiment towards learning",
            "Focus on innovation and technology"
        ],
        "dominant_topics": ["Technology", "Innovation", "Learning"],
        "engagement_level": "High",
        "sentiment_trend": "Positive",
        "recommended_resources": [
            {"resource": "Coursera (Tech Courses)", "priority": "High",
             "reason": "Access to cutting-edge tech courses"},
            {"resource": "Udacity (Nano Degrees)", "priority": "High", "reason": "Specialized tech certifications"},
            {"resource": "Pluralsight", "priority": "Medium", "reason": "Practical tech skill development"}
        ]
    },
    1: {
        "name": "Business & Management Students",
        "characteristics": [
            "Business and management education",
            "Moderate engagement in strategic topics",
            "Neutral to positive learning sentiment",
            "Focus on leadership and strategy"
        ],
        "dominant_topics": ["Management", "Leadership", "Strategy"],
        "engagement_level": "Good",
        "sentiment_trend": "Neutral-Positive",
        "recommended_resources": [
            {"resource": "LinkedIn Learning (Business)", "priority": "High",
             "reason": "Leadership and strategy courses"},
            {"resource": "Coursera (Business)", "priority": "High", "reason": "Data-driven decision-making skills"},
            {"resource": "Harvard Business Review", "priority": "Medium", "reason": "Industry insights and trends"}
        ]
    },
    2: {
        "name": "Creative & Arts Students",
        "characteristics": [
            "Arts and humanities background",
            "Variable engagement patterns",
            "Mixed sentiment about learning",
            "Focus on creativity and expression"
        ],
        "dominant_topics": ["Creativity", "Expression", "Culture"],
        "engagement_level": "Moderate",
        "sentiment_trend": "Mixed",
        "recommended_resources": [
            {"resource": "Skillshare", "priority": "High", "reason": "Creative skill development"},
            {"resource": "Adobe Creative Cloud Tutorials", "priority": "High", "reason": "Industry-standard tools"},
            {"resource": "Domestika", "priority": "Medium", "reason": "Creative project-based learning"}
        ]
    },
    3: {
        "name": "Diverse Learners",
        "characteristics": [
            "Diverse educational backgrounds",
            "Seeking new learning opportunities",
            "Cautious optimism about future",
            "Focus on skill development"
        ],
        "dominant_topics": ["Learning", "Skill Development", "Adaptation"],
        "engagement_level": "Improving",
        "sentiment_trend": "Cautiously Optimistic",
        "recommended_resources": [
            {"resource": "Khan Academy", "priority": "High", "reason": "Broad range of free courses"},
            {"resource": "edX", "priority": "High", "reason": "University-level courses"},
            {"resource": "YouTube Learning Channels", "priority": "Medium", "reason": "Accessible learning content"}
        ]
    }
}


def get_personalized_recommendations(age, education, field, sentiment, cluster_id, text_category):
    """Generate personalized learning recommendations based on profile and text category"""
    cluster_info = CLUSTER_FEATURES.get(cluster_id, CLUSTER_FEATURES[0])
    base_resources = cluster_info["recommended_resources"]

    enhanced_resources = base_resources.copy()

    if sentiment == "Critical" or text_category == "Critical Feedback":
        enhanced_resources.append({
            "resource": "Online Counseling Platforms",
            "priority": "High",
            "reason": "Support for addressing challenges"
        })

    if text_category == "Skill Development Focus":
        enhanced_resources.append({
            "resource": "Specialized Certification Courses",
            "priority": "High",
            "reason": "Targeted skill enhancement"
        })

    if age in ["31–35", "36+"]:
        enhanced_resources.append({
            "resource": "Advanced Leadership Courses",
            "priority": "Medium",
            "reason": "Valuable for experienced learners"
        })

    if text_category == "Positive Feedback":
        enhanced_resources.append({
            "resource": "Advanced Specialization Courses",
            "priority": "Medium",
            "reason": "Build on positive experiences"
        })

    unique_resources = []
    seen_resources = set()
    for resource in enhanced_resources:
        if resource["resource"] not in seen_resources:
            unique_resources.append(resource)
            seen_resources.add(resource["resource"])

    return unique_resources[:4]


# App Configuration
st.set_page_config(page_title="Graduate Intelligence Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
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
    .resource-card {
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

# Main Header
st.markdown("""
<div class="main-header">
    <h1>Workforce Readiness Intelligence Dashboard</h1>
    <p>Comprehensive Analysis: Sentiment • Topics • Recommendations • Learning Guidance</p>
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
        st.markdown("#### Education")
        education = st.selectbox("Education Level",
                                 ["Bachelor's Degree", "Postgraduate Degree", "Diploma", "Other"])
        field_of_study = st.selectbox("Field of Study",
                                      ["Science & Technology", "Business & Management", "Arts", "Other"])
        st.markdown("#### Your Thoughts & Goals")
        feedback_text = st.text_area("Share your feedback about your educational experience:",
                                     placeholder="Tell us about your overall experience, challenges, and thoughts...")
        learning_goals = st.text_area("Describe your learning goals and background:",
                                      placeholder="Describe your field, learning goals, and aspirations...")
        topic_text = st.text_area("Share any additional thoughts or responses:",
                                  placeholder="Any other thoughts, concerns, or topics you'd like to discuss...")
        submitted = st.form_submit_button("🚀 Analyze My Profile", use_container_width=True)

with col2:
    st.markdown("### 📊 Analysis Results")
    if submitted:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [" 📈 Overview", " 💭 Sentiment", " 🧠 Topics", " 🎯 Segment", " 📚 Learning Guidance"])

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

            # Text Category Classification
            combined_text = (feedback_text + " " + learning_goals + " " + topic_text).strip()
            if combined_text:
                text_category, text_confidence = hybrid_text_classification(combined_text, sentiment_model,
                                                                            sentiment_vectorizer)
                results['text_category'] = text_category
                results['text_confidence'] = text_confidence

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
                user_values = [age, gender, education, field_of_study, "Yes"]  # Default employment to Yes
                if len(expected_seg_columns) == len(user_values):
                    user_input = pd.DataFrame([user_values], columns=expected_seg_columns)
                else:
                    user_input = pd.DataFrame([[age, gender, education, field_of_study, "Yes"]],
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
                else:
                    results['cluster'] = 3

            # Display overview metrics
            col_metrics = st.columns(4)
            with col_metrics[0]:
                st.metric("Sentiment", results.get('sentiment', 'N/A'))
            with col_metrics[1]:
                st.metric("Text Category", results.get('text_category', 'N/A'))
            with col_metrics[2]:
                st.metric("Dominant Topic", results.get('topic', 'N/A'))
            with col_metrics[3]:
                st.metric("Cluster", f"Cluster {results.get('cluster', 0)}")

            if 'cluster' in results:
                cluster_id = results['cluster']
                if cluster_id in CLUSTER_FEATURES:
                    cluster_info = CLUSTER_FEATURES[cluster_id]
                    st.markdown(f"""
                    <div class="cluster-info">
                        <h3>🎯 Your Segment: {cluster_info['name']}</h3>
                        <p><strong>Engagement Level:</strong> {cluster_info['engagement_level']}</p>
                        <p><strong>Sentiment Trend:</strong> {cluster_info['sentiment_trend']}</p>
                        <p><strong>Text Category:</strong> {results.get('text_category', 'N/A')}</p>
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
                    'Constructive': "Your feedback shows optimism and satisfaction with your educational journey.",
                    'Critical': "Your feedback indicates some concerns or challenges in your experience.",
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
                    user_input_rec = pd.DataFrame([[age, gender, education, field_of_study, "Yes"]],
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
                    metrics = ['Engagement Level', 'Sentiment', 'Learning Satisfaction', 'Skill Match',
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
            st.markdown("#### 📚 Personalized Learning Guidance")
            if 'cluster' in results and 'text_category' in results:
                cluster_id = results['cluster']
                user_sentiment = results.get('sentiment', 'Neutral')
                text_category = results.get('text_category', 'Uncategorized')
                recommended_resources = get_personalized_recommendations(
                    age, education, field_of_study, user_sentiment, cluster_id, text_category
                )
                st.markdown("### 🚀 Recommended Learning Resources")
                st.markdown(f"*Based on your profile and detected text category: {text_category}*")
                for i, resource in enumerate(recommended_resources, 1):
                    priority_class = f"priority-{resource['priority'].lower()}"
                    st.markdown(f"""
                    <div class="resource-card">
                        <h4>{i}. {resource['resource']} <span class="{priority_class}">({resource['priority']} Priority)</span></h4>
                        <p><strong>Why important:</strong> {resource['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("### 📊 Learning Insights & Action Plan")
                col_insights = st.columns(2)
                with col_insights[0]:
                    st.markdown("#### 💡 Immediate Actions")
                    if user_sentiment == "Critical" or text_category == "Critical Feedback":
                        st.write("🔹 Explore support resources or counseling")
                        st.write("🔹 Identify specific areas for improvement")
                        st.write("🔹 Connect with peers or mentors")
                    else:
                        st.write("🔹 Continue exploring advanced courses")
                        st.write("🔹 Join relevant online communities")
                        st.write("🔹 Build on existing strengths")
                    if text_category == "Skill Development Focus":
                        st.write("🔹 Enroll in targeted skill courses")
                with col_insights[1]:
                    st.markdown("#### 📈 Long-term Strategy")
                    if age in ["18–24", "25–30"]:
                        st.write("🔹 Build foundational expertise")
                        st.write("🔹 Explore diverse learning platforms")
                        st.write("🔹 Gain practical experience through projects")
                    else:
                        st.write("🔹 Deepen specialized knowledge")
                        st.write("🔹 Mentor others in your field")
                        st.write("🔹 Stay updated with industry trends")
                    if education == "Bachelor's Degree":
                        st.write("🔹 Consider pursuing advanced degree")
                st.markdown("#### 🌐 Learning Trends Relevant to You")
                learning_trends = {
                    "Science & Technology": [
                        "AI and machine learning courses in demand",
                        "Cloud computing certifications growing",
                        "Cybersecurity training increasingly relevant"
                    ],
                    "Business & Management": [
                        "Data analytics skills becoming essential",
                        "Leadership courses gaining popularity",
                        "Digital transformation driving new learning needs"
                    ],
                    "Arts": [
                        "Digital content creation courses growing",
                        "UX/UI design training in high demand",
                        "Creative software skills becoming crucial"
                    ],
                    "Other": [
                        "Interdisciplinary skills highly valued",
                        "Continuous learning essential for growth",
                        "Online learning platforms expanding access"
                    ]
                }
                trends = learning_trends.get(field_of_study, learning_trends["Other"])
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
                st.warning("Please complete the analysis in other tabs first to see learning guidance.")

    else:
        st.info("👆 Fill in your information and click 'Analyze My Profile' to see comprehensive results!")
        st.markdown("### 🎯 Sample Segment Profiles & Learning Paths")
        for cluster_id, info in CLUSTER_FEATURES.items():
            with st.expander(f"Cluster {cluster_id}: {info['name']}"):
                col_info, col_resources = st.columns(2)
                with col_info:
                    st.write(f"**Engagement Level:** {info['engagement_level']}")
                    st.write(f"**Sentiment Trend:** {info['sentiment_trend']}")
                    st.write("**Characteristics:**")
                    for char in info['characteristics']:
                        st.write(f"• {char}")
                with col_resources:
                    st.write("**Top Recommended Resources:**")
                    for resource in info['recommended_resources'][:2]:
                        st.write(f"• {resource['resource']} ({resource['priority']} Priority)")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <h4>🎓 Workforce Readiness Intelligence Dashboard</h4>
    <p><strong>Master of Data Analytics - Business Analytics Seminar</strong> | Group 05 </p>
</div>
""", unsafe_allow_html=True)

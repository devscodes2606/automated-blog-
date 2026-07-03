import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime

st.set_page_config(page_title="Blog Content Tracker", layout="wide")

@st.cache_data
def generate_blog_data():
    """Simulates pulling data from Google Analytics/Search Console"""
    np.random.seed(42)
    num_blogs = 100
    
    data = {
        'post_id': range(1, num_blogs + 1),
        'title': [f"Data Science Blog {i}" for i in range(1, num_blogs + 1)],
        'publish_date': [datetime.date.today() - datetime.timedelta(days=np.random.randint(10, 700)) for _ in range(num_blogs)],
        'word_count': np.random.randint(500, 3000, num_blogs),
        'keyword_density': np.random.uniform(0.5, 3.5, num_blogs), 
        'views': np.random.randint(100, 50000, num_blogs),
        'clicks': np.random.randint(10, 5000, num_blogs),
        'bounce_rate': np.random.uniform(30.0, 90.0, num_blogs),
        'category': np.random.choice(['AI', 'Python', 'SEO', 'Data Engineering'], num_blogs)
    }
    
    df = pd.DataFrame(data)
    df.loc[5, 'views'] = np.nan 
    
    return df

def clean_and_engineer_features(df):
    df['views'] = df['views'].fillna(df['views'].median())

    df['publish_date'] = pd.to_datetime(df['publish_date']).dt.normalize()
    today = pd.to_datetime(datetime.date.today())
    df['blog_age_days'] = (today - df['publish_date']).dt.days
    
    df['ctr_proxy'] = (df['clicks'] / df['views']) * 100
    df['engagement_score'] = (df['views'] * 0.4) + (df['clicks'] * 0.6) - (df['bounce_rate'] * 10)
    
    return df

def classify_content(df):
    thresholds = df['engagement_score'].quantile([0.33, 0.66])
    
    def get_class(score):
        if score < thresholds.iloc[0]: return "Low-Performing"
        elif score < thresholds.iloc[1]: return "Average-Performing"
        else: return "High-Performing"
        
    df['performance_tier'] = df['engagement_score'].apply(get_class)
    return df

def train_predictive_model(df):
    """Predicts future views based on content characteristics"""
    features = ['word_count', 'keyword_density', 'blog_age_days', 'bounce_rate']
    X = df[features]
    y = df['views']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    df['predicted_views'] = model.predict(X)
    return df, model

def generate_recommendations(df):
    recommendations = []
    for _, row in df.iterrows():
        if row['performance_tier'] == 'Low-Performing' and row['blog_age_days'] > 180:
            recommendations.append("Needs Update & SEO Audit")
        elif row['bounce_rate'] > 75:
            recommendations.append("Improve Formatting & Internal Links")
        elif row['word_count'] < 800 and row['performance_tier'] != 'High-Performing':
            recommendations.append("Expand Content Length")
        elif row['performance_tier'] == 'High-Performing':
            recommendations.append("Promote & Build Backlinks")
        else:
            recommendations.append("Monitor")
            
    df['action_item'] = recommendations
    return df


def main():
    st.title("📊 Automated Blog Content Tracker & AI Analyzer")
    st.write("This dashboard tracks blog performance, predicts future traffic, and recommends SEO actions.")

    raw_data = generate_blog_data()
    clean_data = clean_and_engineer_features(raw_data.copy())
    classified_data = classify_content(clean_data)
    ml_data, model = train_predictive_model(classified_data)
    final_df = generate_recommendations(ml_data)

    st.header("Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Blogs Tracked", len(final_df))
    col2.metric("Total Views", f"{int(final_df['views'].sum()):,}")
    col3.metric("Avg Bounce Rate", f"{final_df['bounce_rate'].mean():.1f}%")
    col4.metric("High-Performing Blogs", len(final_df[final_df['performance_tier'] == 'High-Performing']))

    st.header("Exploratory Data Analysis")
    tab1, tab2 = st.tabs(["Performance Tiers", "Category Analysis"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=final_df, x='performance_tier', hue='performance_tier', palette='viridis', order=['High-Performing', 'Average-Performing', 'Low-Performing'], legend=False, ax=ax)
        ax.set_title("Distribution of Blog Performance")
        st.pyplot(fig)
        plt.close(fig) 
        
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=final_df, x='category', y='views', hue='category', estimator=np.mean, palette='magma', legend=False, ax=ax)
        ax.set_title("Average Views by Blog Category")
        st.pyplot(fig)
        plt.close(fig)
        
    st.header("AI Action Engine (Recommendations)")
    st.write("Here is what the system recommends for your worst-performing content.")
    
    action_df = final_df[['title', 'views', 'performance_tier', 'action_item']].sort_values(by='views')
    
    st.dataframe(action_df[action_df['action_item'] != 'Monitor'].head(10), use_container_width=True)

    st.header("Predictive Traffic Simulator")
    st.write("Adjust the parameters to see how structural changes might affect a blog's views based on the Random Forest model.")
    
    col_a, col_b, col_c = st.columns(3)
    sim_words = col_a.slider("Word Count", 500, 3000, 1500)
    sim_kwd = col_b.slider("Keyword Density (%)", 0.5, 4.0, 1.5)
    sim_age = col_c.slider("Blog Age (Days)", 0, 700, 100)
    sim_bounce = st.slider("Expected Bounce Rate", 30.0, 90.0, 60.0)
    
    input_features = pd.DataFrame(
        [[sim_words, sim_kwd, sim_age, sim_bounce]], 
        columns=['word_count', 'keyword_density', 'blog_age_days', 'bounce_rate']
    )
    prediction = model.predict(input_features)
    st.success(f"📈 **Predicted Views:** {int(prediction[0]):,}")

    with st.expander("View Full Processed Dataset"):
        st.dataframe(final_df, use_container_width=True)

if __name__ == "__main__":
    main()

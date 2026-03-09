import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# --- 1. Page Config (Set to Wide and Light) ---
st.set_page_config(page_title="World Happiness Insights Dashboard", page_icon="🌍", layout="wide")

# --- 2. Data Loading ---
@st.cache_data
def load_data():
    return pd.read_csv("data/clean_happiness_data.csv")

df = load_data()

# --- 3. Sidebar ---
with st.sidebar:
    st.title("✨ Project Purpose")
    st.markdown("I built this dashboard to analyze world happiness factors.")
    st.divider()
    
    st.header("Global Filters")
    year_choice = st.select_slider("Select a Year", options=sorted(df['year'].unique()))
    selected_countries = st.multiselect("Filter by Countries", sorted(df['country'].unique()))
    
# Filter logic
filtered_df = df[df['year'] == year_choice]
if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

# --- 4. Main Title & Introduction ---
st.title("🌍 World Happiness Analysis (2015-2019)")
st.write(f"In this project, **I am investigating** how factors like money and health impact life satisfaction in **{year_choice}**.")

# --- 5. Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview","💡 Project Core Insight", "🔮 AI Prediction", "🗂️ Dataset Overview"])

# --- TAB 1: OVERVIEW ---
with tab1:
    # High-contrast Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Score", f"{filtered_df['score'].mean():.2f}")
    c1.metric("Standard Deviation", f"{filtered_df['score'].std():.2f}")
    c2.metric("Lowest Score", f"{filtered_df['score'].min():.2f}")
    c2.metric("Highest Score", f"{filtered_df['score'].max():.2f}")
    c3.metric("Sample Size", f"{len(filtered_df)} Countries")

    st.subheader("Global Happiness Map")
    # Using 'Viridis' for maximum visibility/contrast
    fig_map = px.choropleth(filtered_df, locations="country", locationmode='country names', 
                            color="score", hover_name="country", 
                            color_continuous_scale="Viridis")
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")


    st.subheader("📈 Deep Dive Analysis")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Top 10 Happiest Countries")
        top_10 = filtered_df.nlargest(10, 'score')
        fig_top = px.bar(top_10, x='score', y='country', orientation='h', 
                         color='score', color_continuous_scale='Greens', text_auto='.2f')
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, font=dict(color="black"))
        st.plotly_chart(fig_top, use_container_width=True)

    with col_right:
        st.subheader("Bottom 10 Happiest Countries")
        bottom_10 = filtered_df.nsmallest(10, 'score')
        fig_bot = px.bar(bottom_10, x='score', y='country', orientation='h', 
                         color='score', color_continuous_scale='Reds', text_auto='.2f')
        fig_bot.update_layout(yaxis={'categoryorder':'total descending'}, font=dict(color="black"))
        st.plotly_chart(fig_bot, use_container_width=True)
    
    st.divider()
    st.subheader("Correlation Heatmap")
    corr = df[['score', 'gdp', 'social_support', 'life_expectancy', 'freedom', 'corruption']].corr()
    fig_heat = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.subheader("3D Factor Analysis (GDP vs Social Support vs Score)")
    fig_3d = px.scatter_3d(filtered_df, x='gdp', y='social_support', z='score',
                           color='score', size='life_expectancy', hover_name='country',
                           color_continuous_scale='Viridis')
    st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 3: Insight ---
with tab2:
    st.header("💡 My Key Findings")
    st.write("I analyzed the drivers of global happiness from 2015-2019. Here is what I discovered:")

    st.divider()

    # Section 1: Top Drivers
    st.subheader("🚀 The Main Drivers")
    st.markdown("""
    * **Economic Power:** My analysis shows **GDP per Capita** has the strongest link to happiness.
    * **Social Safety Nets:** **Social Support** is the second most critical factor; people are happier when they feel supported by their community.
    * **Health Matters:** I observed a direct linear connection between **Healthy Life Expectancy** and higher scores.
    """)

    st.divider()

    # Section 2: Surprising Results
    st.subheader("🧐 Interesting Observations")
    st.markdown("""
    * **Freedom vs. Corruption:** I found that the **Freedom to make life choices** impacts happiness more significantly than the perception of government corruption.
    * **The Generosity Gap:** Surprisingly, **Generosity** showed the weakest correlation in my model, suggesting that systemic factors (wealth/health) outweigh individual charity in national rankings.
    """)

    st.divider()

    # Section 3: Model Conclusion
    st.subheader("🤖 My Predictor Insights")
    st.write("""
    I built a Linear Regression model that successfully predicts happiness scores using these features. 
    **My conclusion:** Happiness is not random; it is built on a foundation of economic stability, health, and social connection.
    """)

    st.info("**My Final Take:** While money (GDP) is a huge factor, it needs social support and health to truly maximize a nation's happiness.")
# --- TAB 3: PREDICTION ---
with tab3:
    st.header("🔮 Happiness Predictor")
    st.write("I created this tool to simulate how changes in factors affect the happiness score.")
    
    # Model Logic
    features = ['gdp', 'social_support', 'life_expectancy', 'freedom', 'generosity', 'corruption']
    X = df[features].fillna(0)
    y = df['score']
    my_model = LinearRegression().fit(X, y)

    # Input Sliders
    p1, p2, p3 = st.columns(3)
    with p1:
        i_gdp = st.slider("GDP per Capita", 0.0, 2.0, 1.0)
        i_soc = st.slider("Social Support", 0.0, 2.0, 1.0)
    with p2:
        i_health = st.slider("Life Expectancy", 0.0, 1.5, 0.7)
        i_free = st.slider("Freedom", 0.0, 1.0, 0.5)
    with p3:
        i_gen = st.slider("Generosity", 0.0, 1.0, 0.2)
        i_corr = st.slider("Corruption", 0.0, 1.0, 0.1)

    prediction = my_model.predict([[i_gdp, i_soc, i_health, i_free, i_gen, i_corr]])[0]
    
    st.divider()
    # Use st.success for a clean, green-themed high-contrast box
    st.success(f"### Predicted Happiness Score: {prediction:.2f}")

# --- TAB 4: DATASET ---
with tab4:
    st.header("🗂️ Dataset Overview")
    st.write("Here is what each feature represents in the World Happiness data:")
    # Short definitions in a high-contrast box
    st.write("""
    - **😊 Score**: The total happiness rank based on life satisfaction surveys.
    - **💰 GDP per Capita**: The economic output per person (Wealth).
    - **🤝 Social Support**: Having friends or family to count on in times of trouble.
    - **🏥 Life Expectancy**: The average number of healthy years a person lives.
    - **🕊️ Freedom**: The perceived freedom to make your own life choices.
    - **❤️ Generosity**: The average amount of money people donate to charity.
    - **⚖️ Corruption**: The level of trust in government and lack of perceived bribery.
    """)

    st.divider()
    st.subheader("The Cleaned Dataset")
    st.dataframe(df, use_container_width=True)
    st.subheader("Statistical Summary")
    st.write(df.describe())

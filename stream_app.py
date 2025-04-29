import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pymongo import MongoClient
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# Set up the main page configuration (title, layout, icon) 
st.set_page_config(
    page_title="Global Life Expectancy Dashboard",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)
# MongoDB connection and data loading
@st.cache_data(ttl=3600)
def load_and_prepare_data():
    """Load and prepare data from MongoDB"""
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection
        db = client["LifeExpectancyDB"]
        collections = {
            "Change from 2019 to 2023": None,
            "World Health Organization (2019)": None,
            "CIA World Factbook (2022)": None,
            "World Bank Group (2022)": None,
            "Estimate of life expectancy for various ages in 2023": None,
            "OECD (2022)": None
        }
        for name in collections.keys():
            try:
                # Load data from MongoDB
                data = list(db[name].find({}, {'_id': 0}))
                df = pd.DataFrame(data)
                # Basic cleaning
                df = df.dropna(how='all')
                df = df.rename(columns=lambda x: str(x).strip())
                # Standardize country column names
                country_columns = ['Country', 'Countries and territories', 'country', 'Location', 'Entity']
                for col in country_columns:
                    if col in df.columns:
                        df = df.rename(columns={col: 'Country'})
                        break
                # Convert numeric columns
                numeric_cols = ['2023', 'Male', 'Female', 'Sex gap', '2019', '2020', '2021', '2022']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                collections[name] = df
            except Exception as e:
                st.warning(f"Error loading collection {name}: {str(e)}")
                collections[name] = pd.DataFrame()
        return collections
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.stop()
# Load data
all_data = load_and_prepare_data()
life_df = all_data.get("Change from 2019 to 2023", pd.DataFrame())
change_df = all_data.get("World Health Organization (2019)", pd.DataFrame())
# Clean main dataframe
if not life_df.empty:
    life_df = life_df.copy()
    life_df['Country'] = life_df['Country'].astype(str).str.strip()
    life_df = life_df.dropna(subset=['2023', 'Sex gap'])
# Main app
st.title("🌍 Global Life Expectancy Dashboard")
# Create a sidebar to navigate between pages
with st.sidebar:
    page = st.radio("Select Page:", ["Global Overview", "Life Expectancy Trends", "Data Explorer"], index=0)
    if not life_df.empty and 'Region' in life_df.columns:
        regions = life_df['Region'].unique()
        selected_region = st.selectbox("Select Region:", ["All"] + list(regions), index=0)
    else:
        selected_region = "All"
# Page 1: Global Overview
if page == "Global Overview" and not life_df.empty:
    st.header("Global Overview")
 # Show top metrics: highest, lowest life expectancy and average gender gap
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Highest Life Expectancy", 
                 f"{life_df['2023'].max():.1f} years", 
                 life_df.loc[life_df['2023'].idxmax(), 'Country'])
    with col2:
        st.metric("Lowest Life Expectancy", 
                 f"{life_df['2023'].min():.1f} years", 
                 life_df.loc[life_df['2023'].idxmin(), 'Country'])
    with col3:
        st.metric("Average Gender Gap", 
                 f"{life_df['Sex gap'].mean():.1f} years")
    # Draw an interactive map of countries colored by life expectancy in 2023
    st.subheader("🗺️ World Map")
    fig = px.choropleth(
        life_df,
        locations="Country",
        locationmode="country names",
        color="2023",
        hover_name="Country",
        hover_data={"2023":":.1f", "Male":":.1f", "Female":":.1f", "Sex gap":":.1f"},
        color_continuous_scale=px.colors.sequential.Plasma,
        range_color=(life_df["2023"].min(), life_df["2023"].max()),
        projection="natural earth",
        labels={"2023": "Life Expectancy"},
        height=600
    )
    fig.update_geos(showcountries=True, showcoastlines=False, showland=True, landcolor="#a71d31")
    fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
    st.plotly_chart(fig, use_container_width=True)
    # Get top 50 and bottom 50 countries by life expectancy
    st.subheader("Top & Bottom 50 Countries")
    tab1, tab2 = st.tabs(["📈 Top 50", "📉 Bottom 50"])
    with tab1:
        top50 = (life_df.sort_values(by="2023", ascending=False).drop_duplicates(subset=['Country'])
                 .head(50)
                 .reset_index(drop=True))
        st.dataframe(
            top50[['Country', '2023', 'Male', 'Female', 'Sex gap']]
            .style
            .background_gradient(subset=["2023"], cmap="Reds")
            .format({"2023": "{:.1f}", "Male": "{:.1f}", "Female": "{:.1f}", "Sex gap": "{:.1f}"}),
            use_container_width=True,
            height=700
        )
    with tab2:
        bottom50 = (life_df.sort_values(by="2023", ascending=True).drop_duplicates(subset=['Country'])
                   .head(50)
                   .reset_index(drop=True))
        st.dataframe(
            bottom50[['Country', '2023', 'Male', 'Female', 'Sex gap']]
            .style
            .background_gradient(subset=["2023"], cmap="Blues")
            .format({"2023": "{:.1f}", "Male": "{:.1f}", "Female": "{:.1f}", "Sex gap": "{:.1f}"}),
            use_container_width=True,
            height=700
        )
# Apply KMeans clustering to group countries by life expectancy and gender gap
    st.subheader("KMeans Clustering: Life Expectancy vs gender Gap")
    required_cols = ['2023', 'Recovery from COVID-19: 2019:2023', 'Sex gap']
    available_cols = [col for col in required_cols if col in life_df.columns]
    if len(available_cols) < 2:
        st.warning("Not enough required columns for clustering.")
    else:
        selected_cols = st.multiselect(
            "Select columns for clustering:", 
            options=available_cols, 
            default=available_cols[:2]
        )
        n_clusters = st.slider("Number of clusters:", 2, 6, 3)
        try:
            cluster_data = life_df[selected_cols].dropna()
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            clusters = kmeans.fit_predict(cluster_data)
            fig = px.scatter(
                life_df.loc[cluster_data.index],
                x=selected_cols[0],
                y=selected_cols[1],
                color=clusters.astype(str),
                hover_name="Country",
                labels={
                    selected_cols[0]: selected_cols[0],
                    selected_cols[1]: selected_cols[1],
                    "color": "Cluster"
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
                height=600
            )
            fig.update_layout(
                title=f"KMeans Clustering: {selected_cols[0]} vs {selected_cols[1]}",
                xaxis_title=selected_cols[0],
                yaxis_title=selected_cols[1],
                legend_title="Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)
            # Display clusters
            cluster_df = life_df.loc[cluster_data.index].copy()
            cluster_df['Cluster'] = clusters
            cluster_df=cluster_df.drop_duplicates(subset=['Country'])
            cluster_groups = cluster_df.groupby('Cluster')['Country'].apply(list)
            # Create a tidy display of clusters
            max_len = max(len(x) for x in cluster_groups)
            cluster_display = pd.DataFrame({
                f'Cluster {i}': cluster_groups.get(i, []) + ['']*(max_len - len(cluster_groups.get(i, [])))
                for i in range(n_clusters)
            })
            st.dataframe(cluster_display, use_container_width=True)
        except Exception as e:
            st.error(f"Clustering failed: {str(e)}")
# Plot relationship between life expectancy and gender gap
# Page 2: Life Expectancy Trends
elif page == "Life Expectancy Trends" and not life_df.empty:
    st.header("Gender Gap Analysis")
    # Clean data for gender analysis
    gender_df = life_df.dropna(subset=['2023', 'Sex gap']).copy()
    countries = sorted(gender_df['Country'].unique())
    selected_countries = st.multiselect(
        "Select countries to highlight:", 
        options=countries,
        default=[]
    )
    gender_df["Highlight"] = gender_df['Country'].apply(
        lambda x: "Highlighted" if x in selected_countries else "Other"
    )
    # Create scatter plot with absolute values for size
    fig = px.scatter(
        gender_df,
        x="2023",
        y="Sex gap",
        hover_name="Country",
        color="Highlight",
        size=np.abs(gender_df["Sex gap"]),
        trendline="lowess",
        color_discrete_map={
            "Highlighted": "#ffc300",
            "Other": "#2176ff"
        },
        labels={
            "2023": "Life Expectancy (2023)", 
            "Sex gap": "Gender Gap (F-M)"
        },
        height=600
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    # Display top & bottom gender gaps
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📈 Top 30 Gender Gaps**")
        st.dataframe(
            gender_df.drop_duplicates().nlargest(30, "Sex gap")[['Country', 'Sex gap', '2023']],
            use_container_width=True
        )
    with col2:
        st.markdown("**📉 Smallest 30 Gaps**")
        st.dataframe(
            gender_df.drop_duplicates().nsmallest(30, "Sex gap")[['Country', 'Sex gap', '2023']],
            use_container_width=True
        )
    covid_impact = change_df[['Countries', 'All', 'All.1']].rename(
      columns={
        'All': 'At_Birth_2019',
        'All.1': 'At_Birth_2022'
    }
)
# Years for the trend
    years = ['2019', '2020', '2021', '2022', '2023']
# Prepare the data
    if not life_df.empty:
    # Melt the dataframe to long format
        df_long = life_df.melt(
         id_vars=['Country'], 
         value_vars=[col for col in years if col in life_df.columns],
         var_name='Year', 
         value_name='Life Expectancy'
    )
    # Convert year to int
    df_long['Year'] = df_long['Year'].astype(int)
    st.subheader("Life Expectancy Trends by Country")
    available_countries = df_long['Country'].unique().tolist()
    selected_countries = st.multiselect(
        "Select one or more countries:",
        options=available_countries,
        default=available_countries[:2]  
    )
    if selected_countries:
        df_selected = df_long[df_long['Country'].isin(selected_countries)]
        # Create the plot
        fig = px.line(
           df_selected,
           x='Year',
           y='Life Expectancy',
           color='Country',
           markers=True,
           title="Life Expectancy Trend (2019–2023)",
           labels={'Life Expectancy': 'Life Expectancy (Years)', 'Year': 'Year'}
        )
        fig.update_layout(
            xaxis=dict(
                dtick=1,
                showgrid=True,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgrey'
            ),
            plot_bgcolor='white',
            width=1000,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one country to display the chart.")   
# Calculate the decline in life expectancy between 2019 and 2022 
    covid_impact['Decline'] = covid_impact['At_Birth_2019'] - covid_impact['At_Birth_2022']
# Sort countries by the amount of decline 
    top_decline = covid_impact.sort_values('Decline', ascending=False)
# Display it 
    if not top_decline.empty:
        st.subheader("Life Expectancy Changes: 2019 vs 2022")
        selected_countries = st.multiselect(
            "Select countries to compare:",
            options=top_decline['Countries'].unique().tolist(),
            default=['Australia', 'Greece', 'Lesotho', 'Egypt']
    )
        if selected_countries:
            df_selected = top_decline[top_decline['Countries'].isin(selected_countries)]
        
            fig = go.Figure() 
        # Add traces for all selected countries with uniform styling
            for i, country in enumerate(df_selected['Countries']):
                fig.add_trace(go.Scatter(
                    x=[df_selected['At_Birth_2019'].iloc[i], df_selected['At_Birth_2022'].iloc[i]],
                    y=[i, i],
                    mode='lines+markers',
                    line=dict(color='#636EFA', width=2),  
                    marker=dict(size=10),
                    name=country
             ))
                fig.add_trace(go.Scatter(
                    x=[df_selected['At_Birth_2019'].iloc[i]],
                    y=[i],
                    mode='markers',
                    marker=dict(color='#636EFA', size=10, symbol='circle'),
                    name='2019' if i == 0 else None,
                    showlegend=(i == 0)
            ))
                fig.add_trace(go.Scatter(
                    x=[df_selected['At_Birth_2022'].iloc[i]],
                    y=[i],
                    mode='markers',
                    marker=dict(color='#EF553B', size=10, symbol='circle'),
                    name='2022' if i == 0 else None,
                    showlegend=(i == 0)
                ))
            fig.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(df_selected))),
                    ticktext=df_selected['Countries'],
                ),
                xaxis_title='Life Expectancy at Birth (Years)',
                title='Life Expectancy Comparison (2019 vs 2022)',
                plot_bgcolor='white',
                width=1000,
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=100, r=20, t=60, b=40)
            )
        
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one country to display the comparison.")
# Page 3: Data Explorer
elif page == "Data Explorer":
    st.header("📊 Data Explorer")
    # Reconnect to MongoDB here for Data Explorer
    client = MongoClient("mongodb://localhost:27017/")
    db = client["LifeExpectancyDB"]
    collections = db.list_collection_names()
    selected_collection = st.selectbox("Select a table to display:", collections)
    # Display data from the selected collection
    if selected_collection:
        data = list(db[selected_collection].find())
        df = pd.DataFrame(data)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"]).drop_duplicates()
        st.write(f"Displaying `{selected_collection}`:")
        with st.expander("Basic Info"):
            st.write(f"**Number of rows:** {df.shape[0]}")
            st.write(f"**Number of columns:** {df.shape[1]}")
            st.write("**Column Names:**")
            st.write(df.columns.tolist())
            st.write("**Data overview:**")
            st.dataframe(df.describe(include='all'), use_container_width=True)
        st.dataframe(df, use_container_width=True, height=600)
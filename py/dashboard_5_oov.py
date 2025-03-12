import streamlit as st
import pandas as pd
import plotly.express as px

# ---- Constants and Config ----
PARTY_COLORS = {
    'Democrat': '#0015BC',
    'Republican': '#FF0000',
    'Democratic': '#0015BC',
    'Both': '#800080',
    'dem': '#0015BC',
    'rep': '#FF0000'
}

# ---- Page Configuration ----
st.set_page_config(
    page_title="News Analysis Dashboard",
    layout="wide"
)

# ---- Data Loading Functions ----
@st.cache_data
def load_data():
    """Load and preprocess all required data."""
    domain_df = pd.read_csv(r"csv\all_analysis_data.csv")
    posts_df = pd.read_csv(r"csv\scatter_analysis.csv")
    return domain_df, posts_df

def filter_dataframe(df, filters):
    """Apply filters to dataframe."""
    filtered_df = df.copy()

    # Score range filter
    filtered_df = filtered_df[
        (filtered_df["Score"] >= filters["score_range"][0]) &
        (filtered_df["Score"] <= filters["score_range"][1])
    ]

    # Categorical filters
    if filters["party"] != "All":
        filtered_df = filtered_df[filtered_df["Global_Classification"] == filters["party"]]
    if filters["sentiment"] != "All":
        filtered_df = filtered_df[filtered_df["Sentiment"] == filters["sentiment"]]
    if filters["local_cat"] != "All":
        filtered_df = filtered_df[filtered_df["Local_Category"] == filters["local_cat"]]
    else:
        filtered_df = filtered_df[filtered_df["Local_Category"] == "all"]
    if filters["topic"] != "All":
        filtered_df = filtered_df[filtered_df["Topic_Category"] == filters["topic"]]

    return filtered_df

# ---- Visualization Functions ----
def create_party_distribution_chart(data, x_col, y_col="Combination_Total_Count", title=""):
    """Create a bar chart showing distribution by party."""
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        color="Global_Classification",
        color_discrete_map=PARTY_COLORS,
        barmode="group",
        title=title,
        labels={
            x_col: x_col.replace("_", " "),
            y_col: "Number of Articles",
            "Global_Classification": "Party"
        }
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig

def create_score_distribution_chart(data, title=""):
    """Create a box plot showing score distribution."""
    fig = px.box(
        data,
        x="Global_Classification",
        y="Score",
        color="Global_Classification",
        color_discrete_map=PARTY_COLORS,
        points="all",
        title=title,
        labels={
            "Global_Classification": "Party",
            "Score": "Article Score"
        }
    )
    return fig

def calculate_domain_statistics(df):
    """Calculate domain statistics."""
    stats_df = df.groupby("Domain").agg({
        "Combination_Total_Count": "sum",
        "Score": ["mean", "min", "max"],
        "Global_Classification": "first"
    }).round(2)

    stats_df.columns = ["Total Articles", "Average Score", "Min Score", "Max Score", "Party"]
    stats_df = stats_df.reset_index().sort_values("Total Articles", ascending=False)

    # Calculate party-specific counts
    dem_counts = df[df["Global_Classification"].isin(["Democrat", "Democratic", "dem"])].groupby("Domain")[
        "Combination_Total_Count"].sum()
    rep_counts = df[df["Global_Classification"].isin(["Republican", "rep"])].groupby("Domain")[
        "Combination_Total_Count"].sum()

    # Add party-specific counts
    stats_df["Democrat Articles"] = stats_df["Domain"].map(dem_counts).fillna(0).astype(int)
    stats_df["Republican Articles"] = stats_df["Domain"].map(rep_counts).fillna(0).astype(int)

    # Reorder columns
    return stats_df[["Domain", "Total Articles", "Democrat Articles", "Republican Articles",
                     "Average Score", "Min Score", "Max Score", "Party"]]

def show_debug_info(df):
    """Show debug information in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Debug Information")
    st.sidebar.write(f"Total entries: {len(df)}")
    st.sidebar.write(f"Total Articles: {df['Combination_Total_Count'].sum():,}")
    st.sidebar.write(f"Unique Domains: {df['Domain'].nunique()}")

    if st.sidebar.checkbox("Show Data Sample"):
        st.sidebar.dataframe(
            df[["Domain", "Global_Classification", "Combination_Total_Count"]].head()
        )

def run_domain_analysis(df):
    """Run the domain analysis section."""
    st.title("Domain Level Analysis")

    # Sidebar filters
    st.sidebar.header("Filters")
    filters = {
        "score_range": st.sidebar.slider(
            "Score Range",
            min_value=float(df["Score"].min()),
            max_value=float(df["Score"].max()),
            value=(float(df["Score"].min()), float(df["Score"].max())),
            step=0.1
        ),
        "party": st.sidebar.selectbox(
            "Party",
            ["All"] + list(df["Global_Classification"].unique())
        ),
        "sentiment": st.sidebar.selectbox(
            "Sentiment",
            ["All"] + list(df["Sentiment"].unique())
        ),
        "local_cat": st.sidebar.selectbox(
            "Local Category",
            ["All", "local", "non_local"]
        ),
        "topic": st.sidebar.selectbox(
            "Topic",
            ["All"] + list(df["Topic_Category"].unique())
        )
    }

    # Apply filters
    filtered_df = filter_dataframe(df, filters)

    # Domain Deep Dive Section
    st.header("Domain Deep Dive")

    # Topic and Sentiment Distribution
    col1, col2 = st.columns(2)

    with col1:
        topic_dist = filtered_df.groupby(["Topic_Category", "Global_Classification"])[
            "Combination_Total_Count"].sum().reset_index()
        fig = create_party_distribution_chart(
            topic_dist,
            "Topic_Category",
            title="Topic Distribution by Party"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sentiment_dist = filtered_df.groupby(["Sentiment", "Global_Classification"])[
            "Combination_Total_Count"].sum().reset_index()
        fig = create_party_distribution_chart(
            sentiment_dist,
            "Sentiment",
            title="Sentiment Distribution by Party"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Score Analysis
    col3, col4 = st.columns(2)

    with col3:
        fig = create_score_distribution_chart(
            filtered_df,
            title="Score Distribution by Party"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        topic_score = filtered_df.groupby(["Topic_Category", "Global_Classification"])["Score"].mean().reset_index()
        fig = create_party_distribution_chart(
            topic_score,
            "Topic_Category",
            "Score",
            "Average Score by Topic and Party"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Domain Search and Statistics
    st.markdown("---")
    st.subheader("Domain Search")
    col_search1, col_search2 = st.columns([1, 1])

    with col_search1:
        search_text = st.text_input(
            "Search domains by text",
            help="Enter any part of the domain name"
        )

    with col_search2:
        all_domains = sorted(filtered_df["Domain"].unique())
        exact_domain = st.selectbox(
            "Or select exact domain",
            [""] + all_domains,
            help="Select a specific domain from the list"
        )

    # Apply domain filters
    if search_text:
        filtered_df = filtered_df[filtered_df["Domain"].str.contains(search_text, case=False)]
    if exact_domain:
        filtered_df = filtered_df[filtered_df["Domain"] == exact_domain]

    # Domain Statistics
    st.header("Detailed Domain Statistics")
    stats_df = calculate_domain_statistics(filtered_df)

    st.dataframe(
        stats_df,
        use_container_width=True,
        height=400
    )

    # Download button
    st.download_button(
        label="Download Domain Statistics",
        data=stats_df.to_csv(index=False).encode('utf-8'),
        file_name='domain_statistics.csv',
        mime='text/csv'
    )

    # Debug information
    show_debug_info(filtered_df)

def run_posts_analysis(posts_df):
    """Run the posts analysis section of the dashboard."""
    st.title("Articles Analysis")

    # Sidebar filters
    st.sidebar.header("Filters")
    filters = {
        "score_range": st.sidebar.slider(
            "Score Range",
            min_value=float(posts_df["Score"].min()),
            max_value=float(posts_df["Score"].max()),
            value=(float(posts_df["Score"].min()), float(posts_df["Score"].max())),
            step=0.1
        ),
        "party": st.sidebar.selectbox(
            "Party",
            ["All"] + list(posts_df["party"].unique())
        ),
        "sentiment": st.sidebar.selectbox(
            "Sentiment Category",
            ["All"] + list(posts_df["Sentiment_Category"].unique())
        ),
        "local_cat": st.sidebar.selectbox(
            "Local Category",
            ["All"] + sorted(list(posts_df["Local_Category"].dropna().unique()))
        ),
        "topic": st.sidebar.selectbox(
            "Topic Category",
            ["All"] + list(posts_df["Topic_Category"].unique())
        )
    }

    # Apply filters
    filtered_posts = posts_df[
        (posts_df["Score"] >= filters["score_range"][0]) &
        (posts_df["Score"] <= filters["score_range"][1])
    ]

    if filters["party"] != "All":
        filtered_posts = filtered_posts[filtered_posts["party"] == filters["party"]]
    if filters["sentiment"] != "All":
        filtered_posts = filtered_posts[filtered_posts["Sentiment_Category"] == filters["sentiment"]]
    if filters["local_cat"] != "All":
        filtered_posts = filtered_posts[filtered_posts["Local_Category"] == filters["local_cat"]]
    if filters["topic"] != "All":
        filtered_posts = filtered_posts[filtered_posts["Topic_Category"] == filters["topic"]]

    st.subheader("Content Distribution Sunburst")
    st.caption("Hierarchical view of content distribution across different categories")

    # Calculate percentages
    total_count_all = posts_df.shape[0]  # Total data in system
    total_count_filtered = filtered_posts.shape[0]  # Total filtered data

    sunburst_data = filtered_posts.groupby(
        ['Topic_Category', 'Local_Category', 'Sentiment_Category', 'party']
    ).size().reset_index(name='count')

    # Add percentage columns - both for total and filtered data
    sunburst_data['percentage_total'] = (sunburst_data['count'] / total_count_all * 100).round(1)
    sunburst_data['percentage_filtered'] = (sunburst_data['count'] / total_count_filtered * 100).round(1)
    sunburst_data['label'] = (
            sunburst_data['count'].astype(str) +
            '<br>(' + sunburst_data['percentage_filtered'].astype(str) + '% of filtered data' +
            '<br>' + sunburst_data['percentage_total'].astype(str) + '% of total data)'
    )

    fig = px.sunburst(
        sunburst_data,
        path=['Topic_Category', 'Local_Category', 'Sentiment_Category', 'party'],
        values='count',
        title='Content Distribution Hierarchy',
        color='party',
        color_discrete_map=PARTY_COLORS,
        custom_data=['label']
    )

    # Update hover template to show both percentages
    fig.update_traces(
        hovertemplate='<b>%{id}</b><br>' +
                      '%{customdata[0]}<br>' +
                      '<extra></extra>'
    )

    # Make the chart smaller
    fig.update_layout(
        width=600,
        height=600,
        margin=dict(t=30, l=0, r=0, b=0)
    )

    # Add total counts information above the chart
    st.markdown(f"""
    **Total data in system:** {total_count_all:,}  
    **Total data after filtering:** {total_count_filtered:,}
    """)

    st.plotly_chart(fig, use_container_width=True)

    # Additional visualizations in columns
    col1, col2 = st.columns(2)

    with col1:
        # Sentiment vs Score scatter
        st.subheader("Sentiment vs Score")
        fig = px.scatter(
            filtered_posts,
            x="ave_sentiment",
            y="Score",
            color="party",
            color_discrete_map=PARTY_COLORS,
            hover_data=["Local_Category", "Topic_Category"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Party distribution pie chart
        st.subheader("Party Distribution")
        party_fig = px.pie(
            filtered_posts,
            names="party",
            color="party",
            color_discrete_map=PARTY_COLORS
        )
        st.plotly_chart(party_fig, use_container_width=True)

    with col2:
        # Topic distribution by party
        st.subheader("Topic Distribution by Party")
        topic_fig = px.histogram(
            filtered_posts,
            x="Topic_Category",
            color="party",
            barmode="group",
            color_discrete_map=PARTY_COLORS
        )
        topic_fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(topic_fig, use_container_width=True)

        # Summary statistics by party
        st.subheader("Summary Statistics by Party")
        party_stats = filtered_posts.groupby("party").agg({
            "Score": ["count", "mean"],
            "ave_sentiment": "mean"
        }).round(2)
        party_stats.columns = ["Total Articles", "Average Score", "Average Sentiment"]
        st.dataframe(party_stats, use_container_width=True)

    # Debug information
    st.sidebar.markdown("---")
    st.sidebar.write(f"Articles in view: {len(filtered_posts)}")

# ---- Main Dashboard Function ----
def run_dashboard():
    """Main function to run the dashboard."""
    # Load data
    domain_df, posts_df = load_data()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Analysis Type", ["Domain Analysis", "Articles Analysis"])

    if page == "Domain Analysis":
        run_domain_analysis(domain_df)
    else:
        run_posts_analysis(posts_df)

# ---- Run Dashboard ----
if __name__ == "__main__":
    run_dashboard()

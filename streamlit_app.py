import pandas as pd
import plotly.express as px
import streamlit as st # Import Streamlit
import numpy as np  # Library for numerical computations
import seaborn as sns  # Library for data visualization
import matplotlib.pyplot as plt  # Library for creating static and interactive plots

#===============================================================================================

# Load the Fortune 1000 dataset from a CSV file
fortune = pd.read_csv("fortune1000.csv")

st.set_page_config(
    page_title="Robin's Streamlit App",
    page_icon=":smiley:",
    layout="wide",
    #initial_sidebar_state="expanded"
)


# Displaying my name(robin xavier) over the layout-----------------------------
col1, _ = st.columns([0.1, 10])

with col1:
   # using HTML inside, to make layout and display adjustments------------------------------
    st.markdown(
        "<p style='font-size: 16px; margin-top: 0px; margin-bottom: 0px; white-space: nowrap;'>"
        "ROBIN XAVIER (<a href='https://www.linkedin.com/in/robin-xavier-367777217/' target='_blank' style='font-size: 14px;'>linkedin</a>)"
        "</p>",
        unsafe_allow_html=True
    )
#===============================================================================================


st.title("FORTUNE 500 COMPANIES 📈")  # Giving title to the Analysis
st.markdown("Welcome to My Fortune 500 Companies Analysis page! 📊 This interactive dashboard provides insights into the top 500 companies in the world, ranked by revenue. Explore trends, visualize data, and discover key statistics about these industry leaders. Let's dive in! 🚀")
# Sub heading
st.markdown("<h1 style='font-size: 30px;'>Exploratory Data Analysis of Fortune 500 Companies with Interactive Visualizations 🐼</h1>", unsafe_allow_html=True)

# print Sample data from Fortune data set
st.write("--- Fortune Sample Data (first 5 rows) ---")
# Display the first few rows of the dataset to verify its contents
st.dataframe(fortune.head(5))  # first 5 rows of the data set

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Displaying the summary of the fortune describing central tendency and variability
st.write("--- Descriptive (summary) statistics of Fortune 500 ---")
st.dataframe(fortune.describe())
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Displaying the summary of the fortune describing central tendency and variability
st.write("---Transpose Statistical Summary of Fortune 500---")
st.dataframe(fortune.describe().T)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --- Random 20 Rows---
#st.header("")
st.write("---20 Random Fortune 500 Companies---")
st.dataframe(fortune.sample(n=20, axis=0))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#st.markdown("<h1 style='font-size: 25px;'>---Distribution of Fortune 500 Company Profits--Histogram---</h1>", unsafe_allow_html=True)
st.header("Exploring Fortune 500 Companies Through Interactive Visualizations🐼")     #-------------------------------Heading before visualizations.
st.write("---Fortune 500 Companies Ranked by Revenue---")
top = fortune.nlargest(10, 'Revenue')  #--finding or sorting rows based on revenue, first 10.
# Create a bar chart
fig = px.bar(top, x='Company', y='Revenue', title='Top 10 Companies by Revenue')
fig.update_layout(xaxis_tickangle=-45)  # Rotate x-axis labels for better readability

                    # ---Showing the chart using streamlit.
st.plotly_chart(fig, use_container_width=True)     # --allowing the container width adjust with the figure area
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Distribution of Profits Among Fortune 500 Companies---</h1>", unsafe_allow_html=True)

st.write("Using Histogram")# ----------------------------Heading before visualization.
#------------------- Create a histogram to visualize the distribution of profits
fig, ax = plt.subplots(figsize=(4,3))  # Set the figure size
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
sns.histplot(fortune['Profits'], bins=20, kde=True, ax=ax)  # Plot histogram with 20 bins and kernel density estimate
ax.set_title('Distribution of Company Profits', fontsize=7)  # Add title with suitable font size
ax.set_xlabel('Profits', fontsize=6)  # Label x-axis with suitable font size
ax.set_ylabel('Frequency', fontsize=6)  # Label y-axis with suitable font size
ax.tick_params(axis='x', labelsize=5)  # Adjust x-axis tick font size
ax.tick_params(axis='y', labelsize=5)  # Adjust y-axis tick font size
fig.tight_layout()  # Ensure labels fit within the figure area
st.pyplot(fig, use_container_width=False)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Number of Fortune 500 Companies by Sector---</h1>", unsafe_allow_html=True)
st.write("Using Bar Graphs")
#Count the number of companies in each sector/industry
sector_count = fortune['Sector'].value_counts()  # Get the count of companies in each sector

fig, ax = plt.subplots(figsize=(5,4))  # Set figure size
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
sector_count.plot(kind='bar', ax=ax)  # Plot bar chart
ax.set_title('Number of Companies by Sector', fontsize=9)  # Add title with suitable font size
ax.set_xlabel('Sector', fontsize=6)  # Label x-axis with suitable font size
ax.set_ylabel('Count', fontsize=6)  # Label y-axis with suitable font size
ax.tick_params(axis='x', rotation=70, labelsize=4)  # Rotate x-axis labels and adjust font size
ax.tick_params(axis='y', labelsize=5)  # Adjust y-axis tick font size
fig.tight_layout()  # Ensure labels fit within the figure area
st.pyplot(fig, use_container_width=False)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Top 10 Most Common States for Fortune 500 Company Headquarters---</h1>", unsafe_allow_html=True)
st.write("Using Bar Graph")
#------------------most common state for company headquarters
#extract the state from location coloumn
fortune['State'] = fortune['Location'].apply(lambda x: x.split(', ')[1])
#count the number of companies in each state
count = fortune['State'].value_counts().nlargest(10)
#find the most common state

# Create a bar chart
fig, ax = plt.subplots(figsize=(4,3))
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
count.plot(kind='bar', ax=ax)
ax.set_title('Top 10 States by Number of Company Headquarters', fontsize=9)
ax.set_xlabel('State', fontsize=6)
ax.set_ylabel('Number of Companies', fontsize=6)
ax.tick_params(axis='x', rotation=40, labelsize=5)
ax.tick_params(axis='y', labelsize=5)
fig.tight_layout()
st.pyplot(fig, use_container_width=False)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---This analysis explores the impact of Revenue on Profits for Fortune 500 Companies---</h1>", unsafe_allow_html=True)
st.write("Using ScatterPlots")
# Create a scatter plot to visualize the relationship
fig, ax = plt.subplots(figsize=(4,3))  
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
ax.scatter(fortune['Revenue'], fortune['Profits'])  
ax.set_xlabel('Revenue', fontsize=6)  
ax.set_ylabel('Profits', fontsize=6)  
ax.set_title('Relationship between Revenue and Profits', fontsize=9)  
ax.tick_params(axis='x', labelsize=5, rotation=45)
ax.tick_params(axis='y', labelsize=5)
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>Uncovering the Relationship Between Revenue and Rank: Insights from Fortune 500 Companies</h1>", unsafe_allow_html=True)
st.write("Using ScatterPlots")
# Create a scatter plot to visualize the relationship
fig, ax = plt.subplots(figsize=(4,3))  
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
ax.scatter(fortune['Rank'], fortune['Revenue'])  
ax.set_xlabel('Company Rank', fontsize=6)  
ax.set_ylabel('Revenue', fontsize=6)  
ax.set_title('Relationship between Company Rank and Revenue', fontsize=9)  
ax.tick_params(axis='x', labelsize=5, rotation=45)
ax.tick_params(axis='y', labelsize=5)
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Distribution of Profits Across Sectors: An Analysis of Fortune 500 Companies---</h1>", unsafe_allow_html=True)
st.write("Using BoxPlots")
# Create a box plot to visualize the distribution
fig, ax = plt.subplots(figsize=(6,4))  
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
sns.boxplot(x='Sector', y='Profits', data=fortune, ax=ax)  
ax.set_title('Distribution of Profits across Different Sectors', fontsize=9)  
ax.tick_params(axis='x', labelsize=5, rotation=90)
ax.tick_params(axis='y', labelsize=5)
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Is There a Connection Between Revenue, Profits, Employees, and Rank? Let's Take a Closer Look!---</h1>", unsafe_allow_html=True)
st.write("Using Heatmap")

st.write("---Heatmap of Correlations: Revenue, Profits, Employees, and Rank---")
# Calculate the correlation matrix
numeric_cols = ['Revenue', 'Profits', 'Employees', 'Rank']  
fortune[numeric_cols] = fortune[numeric_cols].apply(pd.to_numeric, errors='coerce')  
corr_matrix = fortune[numeric_cols].corr()  

# Create a heatmap to visualize correlations
fig, ax = plt.subplots(figsize=(4,3))  
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, ax=ax, annot_kws={"fontsize":5})  
ax.set_title('Heatmap of Correlations', fontsize=9)  
ax.tick_params(axis='x', labelsize=5, rotation=45)
ax.tick_params(axis='y', labelsize=5)
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Let's Analyze Average Revenue and Profit per Sector---</h1>", unsafe_allow_html=True)
st.write("Using Bar Graph")

# Calculate the average revenue and profit per sector and select top 5
sector_average = fortune.groupby('Sector')[['Revenue', 'Profits']].mean().reset_index().sort_values(by='Revenue', ascending=False).head(5)

# Create a grouped bar chart to visualize the comparison
x = np.arange(len(sector_average))  
fig, ax = plt.subplots(figsize=(4,3))
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
bar_width = 0.35  

# Plot revenue and profit bars
ax.bar(x - bar_width/2, sector_average['Revenue'], bar_width, label='Revenue')
ax.bar(x + bar_width/2, sector_average['Profits'], bar_width, label='Profits')

# Add labels and title
ax.set_xlabel('Sector', fontsize=6)
ax.set_ylabel('Average Value', fontsize=6)
ax.set_title('Average Revenue and Profit per Sector (Top 5)', fontsize=9)

# Set x-axis ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(sector_average['Sector'], rotation=45, fontsize=5)

# Add legend and ensure labels fit within the figure area
ax.legend(fontsize=5)
ax.tick_params(axis='y', labelsize=5)
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Let's Take a Look at the Top 5 Industries with the Highest Average Profit Margin---</h1>", unsafe_allow_html=True)
st.write("Using Bar Graph")
# Calculate average revenue and profits per sector
df = pd.DataFrame(fortune, columns=['Sector', 'Revenue', 'Profits'])
sector_avg = df.groupby('Sector')[['Revenue', 'Profits']].mean().reset_index()
sector_avg['Profit Margin'] = sector_avg['Profits'] / sector_avg['Revenue']

# Sort by profit margin in descending order and select top 5
sector_avg = sector_avg.sort_values(by='Profit Margin', ascending=False).head(5)

# Visualize the top 5 industries by average profit margin
fig, ax = plt.subplots(figsize=(4,3))  
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
ax.bar(sector_avg['Sector'], sector_avg['Profit Margin'])  
ax.set_title('Top 5 Industries by Average Profit Margin', fontsize=9)  
ax.set_xlabel('Sector', fontsize=6)  
ax.set_ylabel('Profit Margin', fontsize=6)  
ax.tick_params(axis='x', labelsize=5, rotation=45)
ax.tick_params(axis='y', labelsize=5)
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)

# Display the results in Streamlit
st.write(sector_avg)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Let's Identify Outliers in Revenue and Profit Using Box Plots---</h1>", unsafe_allow_html=True)
st.write("Using Box Plot")
# Calculate Q1, Q3, and IQR for revenue and profit
df = pd.DataFrame(fortune, columns=['Revenue', 'Profits'])
Q1_revenue = df['Revenue'].quantile(0.25)  
Q3_revenue = df['Revenue'].quantile(0.75)  
IQR_revenue = Q3_revenue - Q1_revenue  

Q1_profit = df['Profits'].quantile(0.25)  
Q3_profit = df['Profits'].quantile(0.75)  
IQR_profit = Q3_profit - Q1_profit  

# Calculate outlier bounds using 1.5*IQR rule
lower_bound_revenue = Q1_revenue - 1.5 * IQR_revenue  
upper_bound_revenue = Q3_revenue + 1.5 * IQR_revenue  

lower_bound_profit = Q1_profit - 1.5 * IQR_profit  
upper_bound_profit = Q3_profit + 1.5 * IQR_profit  

# Identify outliers
outliers_revenue = df[(df['Revenue'] < lower_bound_revenue) | (df['Revenue'] > upper_bound_revenue)]  
outliers_profit = df[(df['Profits'] < lower_bound_profit) | (df['Profits'] > upper_bound_profit)]  

# Create box plots to visualize revenue and profit distributions
fig, ax = plt.subplots(1, 2, figsize=(8,3))  

ax[0].boxplot(df['Revenue'], vert=False)
ax[0].set_title('Revenue Distribution', fontsize=7)  
ax[0].set_facecolor('#E8E8E8')  # Set face color for revenue plot

ax[1].boxplot(df['Profits'], vert=False)
ax[1].set_title('Profit Distribution', fontsize=7)  
ax[1].set_facecolor('#E8E8E8')  # Set face color for profit plot

fig.patch.set_facecolor('#E8E8E8')  # Set face color for figure
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Let's Have a Look at Sectors With the Most Variability in Fortune 500 Company Profits---</h1>", unsafe_allow_html=True)
st.write("Using Bar Graph")

# Calculate the variance of profits for each sector
df = pd.DataFrame(fortune, columns=['Sector', 'Profits'])
sector_variability = df.groupby('Sector')['Profits'].var().reset_index()
sector_variability.columns = ['Sector', 'Variance']
sector_variability = sector_variability.sort_values(by='Variance', ascending=False).reset_index(drop=True)

# Display the result
st.write("Sectors with the most variability in company profits:")
st.write(sector_variability)

# Visualize the variance of profits for each sector
fig, ax = plt.subplots(figsize=(5,4))  
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
ax.bar(sector_variability['Sector'], sector_variability['Variance'])  
ax.set_title('Variance of Profits Across Sectors', fontsize=9)  
ax.set_xlabel('Sector', fontsize=6)  
ax.set_ylabel('Variance', fontsize=6)  
ax.tick_params(axis='x', labelsize=5, rotation=90)
ax.tick_params(axis='y', labelsize=5)
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))  # Format y-axis
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)
print(sector_variability)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate the variance of profits for each sector
df = pd.DataFrame(fortune, columns=['Sector', 'Profits'])
sector_variability = df.groupby('Sector')['Profits'].var().reset_index()
sector_variability.columns = ['Sector', 'Variance']
sector_variability = sector_variability.sort_values(by='Variance', ascending=False).head(10)

# Create a bar chart
st.write("Same But in Streamlit Style")
fig = px.bar(sector_variability, x='Sector', y='Variance', title='Top 10 Sectors by Profit Variance')
fig.update_layout(xaxis_tickangle=-45)  
st.plotly_chart(fig, use_container_width=True)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Fortune 500 Companies with Negative Profits---</h1>", unsafe_allow_html=True)

# Select relevant columns
df = pd.DataFrame(fortune, columns=['Sector', 'Profits', 'State'])

# Filter companies with negative profits
negative_profit_companies = df[df['Profits'] < 0]

# Calculate the proportion of companies with negative profits in each industry
industry_proportion = (negative_profit_companies['Sector'].value_counts() / df['Sector'].value_counts()).sort_values(ascending=False).fillna(0)

# Calculate the proportion of companies with negative profits in each state
state_proportion = (negative_profit_companies['State'].value_counts() / df['State'].value_counts()).sort_values(ascending=False).fillna(0)

st.write("Industry Proportion of Companies with Negative Profits:")
st.write(industry_proportion)

st.write("State Proportion of Companies with Negative Profits:")
st.write(state_proportion)

# Visualize the industry proportion of companies with negative profits
fig, ax = plt.subplots(figsize=(5,4))  
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
industry_proportion.head(10).plot(kind='bar', ax=ax)  
ax.set_title('Industry Proportion of Companies with Negative Profits', fontsize=9)  
ax.set_xlabel('Industry', fontsize=6)  
ax.set_ylabel('Proportion', fontsize=6)  
ax.tick_params(axis='x', labelsize=5, rotation=90)
ax.tick_params(axis='y', labelsize=5)
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)

# Visualize the state proportion of companies with negative profits
fig, ax = plt.subplots(figsize=(5,4))  
fig.patch.set_facecolor('#E8E8E8')  
ax.patch.set_facecolor('#E8E8E8')
state_proportion.head(10).plot(kind='bar', ax=ax)  
ax.set_title('State Proportion of Companies with Negative Profits', fontsize=9)  
ax.set_xlabel('State', fontsize=6)  
ax.set_ylabel('Proportion', fontsize=6)  
ax.tick_params(axis='x', labelsize=5, rotation=90)
ax.tick_params(axis='y', labelsize=5)
fig.tight_layout()  
st.pyplot(fig, use_container_width=False)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 25px;'>---Pair Plot for Numeric Features, Analysing Revenue, Profits and Rank Correlation---</h1>", unsafe_allow_html=True)

# Select relevant numeric columns
df = pd.DataFrame(fortune, columns=['Revenue', 'Profits', 'Rank'])

# Create a pair plot to visualize relationships between numeric features
fig = sns.pairplot(df, height=2, aspect=1.2, plot_kws={'alpha': 0.5, 's': 10})  
st.pyplot(fig.fig, use_container_width=False)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.markdown("<h1 style='font-size: 40px;'>---THANKS!!!---</h1>", unsafe_allow_html=True)
# Displaying my name(robin xavier) over the layout-----------------------------
col1, _ = st.columns([0.1, 10])

with col1:
   # using HTML inside, to make layout and display adjustments------------------------------
    st.markdown(
        "<p style='font-size: 16px; margin-top: 0px; margin-bottom: 0px; white-space: nowrap;'>"
        "ROBIN XAVIER (<a href='https://www.linkedin.com/in/robin-xavier-367777217/' target='_blank' style='font-size: 14px;'>linkedin</a>)"
        "</p>",
        unsafe_allow_html=True
    )
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_csv("/home/robin/Desktop/DATA SCIENTIST/RP2/Assignments/Statistics/EDA assignment 1/fortune1000.csv")

st.title("My EDA Dashboard")

# Select box for column filter
selected_column = st.selectbox("Choose column to visualize", df.columns)

# Show pairplot or scatter
fig = px.histogram(df, x=selected_column)
st.plotly_chart(fig)

# Optional: add more interactive plots
import matplotlib.pyplot as plt

st.subheader("📊 Negative Profit Analysis: Industries and States")

# Filter companies with negative profits
negative_profit_companies = df[df['Profits'] < 0]

# Calculate proportions
industry_proportion = (negative_profit_companies['Sector'].value_counts() / df['Sector'].value_counts()).sort_values(ascending=False).fillna(0)
state_proportion = (negative_profit_companies['State'].value_counts() / df['State'].value_counts()).sort_values(ascending=False).fillna(0)

# 📈 Industry Proportion Plot
st.write("### Industry Proportion of Companies with Negative Profits")
fig1, ax1 = plt.subplots(figsize=(12, 6))
industry_proportion.plot(kind='bar', ax=ax1)
ax1.set_title('Industry Proportion of Companies with Negative Profits')
ax1.set_xlabel('Industry')
ax1.set_ylabel('Proportion')
plt.xticks(rotation=90)
plt.tight_layout()
st.pyplot(fig1)

# 📉 State Proportion Plot
st.write("### State Proportion of Companies with Negative Profits")
fig2, ax2 = plt.subplots(figsize=(12, 6))
state_proportion.plot(kind='bar', ax=ax2)
ax2.set_title('State Proportion of Companies with Negative Profits')
ax2.set_xlabel('State')
ax2.set_ylabel('Proportion')
plt.xticks(rotation=90)
plt.tight_layout()
st.pyplot(fig2)


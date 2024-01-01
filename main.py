import pandas as pd
import numpy as np
import csv
import requests
import base64
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

#from sklearn.feature_extraction.text import TfidfVectorizer
#import spacy
#from gensim import corpora, models
#from gensim.models import Word2Vec
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, Embedding
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from sklearn.model_selection import train_test_split

class job_market:
  def __init__(self, api_key):
    self.api_key=api_key
    self.encoded_api_key = base64.b64encode(f"{api_key}:".encode()).decode()
    self.headers = {'Authorization': f'Basic {self.encoded_api_key}'}

  def get_data(self, job):
    url = f'https://www.reed.co.uk/api/1.0/search?keywords={job}'
    response = requests.get(url, headers=self.headers)
    data_json = response.json()
    data_df = pd.DataFrame(data_json['results'])
    # Ensure 'date' column is in datetime format
    data_df['date'] = pd.to_datetime(data_df['date'])

    # Calculate the date 3 months ago from today
    a_month_ago = datetime.now() - timedelta(days=30)

    # Filter out posts older than 3 months
    data_df = data_df[data_df['date'] > a_month_ago].reset_index()

    return data_df
    return data_df

  def job_stats(self, data_df):
    total_job = len(data_df)
    job_per_day = total_job/30
    # calculae application per job (need to be posted at least 2 week)
    two_weeks_ago = datetime.now() - timedelta(days=14)

    # Filter out posts older than 3 months
    qualified_jobs = data_df[data_df['date'] <= two_weeks_ago]
    application_per_job = np.mean(qualified_jobs['applications'])
    return total_job, job_per_day, application_per_job

  def plot_jobs_by_date(self, data_df, aggregation='day'):
    # Ensure 'date' column is in datetime format
    data_df['date'] = pd.to_datetime(data_df['date'])

    if aggregation == 'day':
      data_grouped = data_df.groupby(data_df['date'].dt.date).size()
    elif aggregation == 'week':
      # Calculate the week commencing (Monday) date for each entry
      data_df['week_commencing'] = data_df['date'].dt.to_period('W').apply(lambda x: x.start_time)
      data_grouped = data_df.groupby('week_commencing').size()
    else:
      raise ValueError("Invalid aggregation: choose 'day' or 'week'")
    # Sort the grouped data
    data_grouped = data_grouped.sort_index()
    # Plotting
    plt.figure(figsize=(12, 6))
    data_grouped.plot(kind='line', marker='o', color='green')
    plt.title(f'Number of Job Posts by {aggregation.title()}')
    plt.xlabel(aggregation.title())
    plt.ylabel('Number of Job Posts')
      # Annotate each point with its value
    for x, y in data_grouped.items():
        label = "{:.0f}".format(y)
        plt.annotate(label,  # this is the text
                      (x, y),  # this is the point to label
                      textcoords="offset points",  # how to position the text
                      xytext=(0,10),  # distance from text to points (x,y)
                      ha='center')  # horizontal alignment can be left, right or center
    plt.xticks(rotation=45)
    st.pyplot(plt)
  def plot_salary_ranges(self, data_df):
    # Remove NaN values and reset index
    data_df = data_df.dropna(subset=['minimumSalary', 'maximumSalary']).reset_index(drop=True)

    # Optionally, handle outliers here...

    # Plotting
    plt.figure(figsize=(12, 6))

    # Scatter plot for min and max salaries
    plt.scatter(data_df.index, data_df['minimumSalary'], color='blue', label='Minimum Salary')
    plt.scatter(data_df.index, data_df['maximumSalary'], color='red', label='Maximum Salary')

    # Connect min and max salaries with a line for each job
    for idx in data_df.index:
        plt.plot([idx, idx], [data_df.loc[idx, 'minimumSalary'], data_df.loc[idx, 'maximumSalary']], color='gray')

    # Styling
    plt.title('Salary Ranges for Job Listings')
    plt.xlabel('Job Listing Index')
    plt.ylabel('Salary')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


api_key = '646bb8ba-a4bf-4d22-b895-b62fdc8a2996'


# Initialize your class (replace 'your_api_key' with your actual key)
job_api = job_market('646bb8ba-a4bf-4d22-b895-b62fdc8a2996')


def main():
    st.title("Reed Jobs Analysis Tool")

    # Stage 1: Fetch Data Based on Job Title
    job_title = st.text_input("Enter Job Title")
    if st.button("Click to get data"):
        data_df = job_api.get_data(job_title)
        if not data_df.empty:
            st.write("Data Fetched Successfully")


            # Display job stats
            total_job, job_per_day, application_per_job = job_api.job_stats(data_df)
            st.write(f"Total Jobs: {total_job}, Jobs per Day: {job_per_day:.2f}, Applications per Job: {application_per_job:.2f}")
            st.write(f"Jobs counts in last 30 days. Number of applications is only considered jobs that have been posted for more than 2 weeks")

            with st.expander("View Number of Jobs Plot"):
              aggregation = st.radio("Select Aggregation Type", ('day', 'week'))
              job_api.plot_jobs_by_date(data_df, aggregation)
  
            with st.expander("View Salary Ranges"):  
              job_api.plot_salary_ranges(data_df)
            
            location = st.text_input("Enter Location")
            if location:
                filtered_data = data_df[data_df['locationName'].str.contains(location, case=False)]
                st.write(filtered_data[['jobTitle', 'minimumSalary', 'maximumSalary', 'employerName', 'applications', 'jobUrl']])

        else:
            st.write("No data found for this job title.")

if __name__ == "__main__":
    main()


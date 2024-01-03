import pandas as pd
import numpy as np
import csv
import requests
import base64
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
from bs4 import BeautifulSoup


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
    #for i in data_df['jobId']:
      #url = f'https://www.reed.co.uk/api/1.0/jobs/{i}'
      #response = requests.get(url, headers=self.headers)
      #data_json = response.json()
      #job_description = data_json['jobDescription']

      #soup = BeautifulSoup(job_description, 'html.parser')
      #job_description = soup.get_text()
      #data_df.loc[data_df.jobId==i,'jobDescription'] = job_description

    return data_df

  def job_stats(self, data_df):


    # Ensure 'date' column is in datetime format
    data_df['date'] = pd.to_datetime(data_df['date'], format='%d/%m/%Y')

    # Calculate the date 3 months ago from today
    a_month_ago = datetime.now() - timedelta(days=30)

    # Filter out posts older than 3 months
    data_df = data_df[data_df['date'] > a_month_ago].reset_index()
    total_job = len(data_df)
    job_per_day = total_job/30

    # calculae application per job (need to be posted at least 2 week)
    two_weeks_ago = datetime.now() - timedelta(days=14)

    # Filter out posts older than 3 months
    qualified_jobs = data_df[data_df['date'] <= two_weeks_ago]
    application_per_job = np.mean(qualified_jobs['applications'])
    med_min=np.median(data_df['minimumSalary'].dropna())
    med_max=np.median(data_df['maximumSalary'].dropna())
    return total_job, job_per_day, application_per_job, med_min, med_max

  def plot_jobs_by_date(self, data_df, aggregation='day'):
    # Ensure 'date' column is in datetime format
    data_df['date'] = pd.to_datetime(data_df['date'], format='%d/%m/%Y')

    # Calculate the date 3 months ago from today
    a_month_ago = datetime.now() - timedelta(days=30)

    # Filter out posts older than 3 months
    data_df = data_df[data_df['date'] > a_month_ago].reset_index()

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

    # Remover outliers
    Q1_min = data_df['minimumSalary'].quantile(0.25)
    Q3_min = data_df['minimumSalary'].quantile(0.75)
    IQR_min = Q3_min - Q1_min
    lower_bound_min = Q1_min - 1.5 * IQR_min
    upper_bound_min = Q3_min + 1.5 * IQR_min
    data_df = data_df[(data_df['minimumSalary'] >= lower_bound_min) & (data_df['minimumSalary'] <= upper_bound_min)]

    Q1_max = data_df['maximumSalary'].quantile(0.25)
    Q3_max = data_df['maximumSalary'].quantile(0.75)
    IQR_max = Q3_max - Q1_max
    lower_bound_max = Q1_max - 1.5 * IQR_max
    upper_bound_max = Q3_max + 1.5 * IQR_max
    data_df = data_df[(data_df['maximumSalary'] >= lower_bound_max) & (data_df['maximumSalary'] <= upper_bound_max)]


    # Plotting
    plt.figure(figsize=(12, 6))

    # Scatter plot for min and max salaries
    plt.scatter(data_df.index, data_df['minimumSalary'], color='blue', label='Minimum Salary')
    plt.scatter(data_df.index, data_df['maximumSalary'], color='green', label='Maximum Salary')

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
  def compare_salary(self, user_salary, data_df, location=None):
    # Filter data by location if specified
    if location:
        data_df = data_df[data_df['locationName'] == location]

    # Merge min and max salaries into a single range
    sal_df=data_df[['minimumSalary','maximumSalary']]
    sal_df=sal_df.dropna()
    all_salaries = []
    for _, row in sal_df.iterrows():
        all_salaries.extend(list(range(int(row['minimumSalary']), int(row['maximumSalary']) + 1)))

    # Calculate percentiles
    percentiles = np.percentile(all_salaries, list(range(0, 101)))
    user_percentile = np.searchsorted(percentiles, user_salary)

    comparison_result = f"Your salary is in the top {100 - user_percentile}% of the market."

    return comparison_result, 100 - user_percentile

api_key = '646bb8ba-a4bf-4d22-b895-b62fdc8a2996'




# Initialize your class (replace 'your_api_key' with your actual key)
job_api = job_market('646bb8ba-a4bf-4d22-b895-b62fdc8a2996')


def main():
    st.title("Reed Jobs Analysis Tool")
    st.markdown("By Anh")

    if 'data_df' not in st.session_state:
        st.session_state['data_df'] = None

    job_title = st.text_input("Enter Job Title")
    if st.button("Click to get data"):
        loading_message = st.markdown("Fetching job data, this might take a while...")
        data_df = job_api.get_data(job_title)
        loading_message.empty()
        if not data_df.empty:
            st.session_state['data_df'] = data_df
            st.write("Data Fetched Successfully")
        else:
            st.write("No data found for this job title.")

    if st.session_state['data_df'] is not None:
        data_df = st.session_state['data_df']

        # Display job stats
        total_job, job_per_day, application_per_job, med_min, med_max = job_api.job_stats(data_df)
        st.write(f"Total Jobs: {total_job}, Jobs per Day: {job_per_day:.2f}, Applications per Job: {application_per_job:.2f}, Typical salary range: £{med_min: .0f} to £{med_max: .0f}")
        st.write(f"Jobs counts in last 30 days. Number of applications per job is only considered jobs that have been posted for more than 2 weeks")

        with st.expander("View Number of Jobs Plot"):
            selected_aggregation = st.session_state.get('aggregation', 'day')
            aggregation = st.radio("Select Aggregation Type", ('day', 'week'), index=('day', 'week').index(selected_aggregation))
            st.session_state['aggregation'] = aggregation
            job_api.plot_jobs_by_date(data_df, aggregation)

        with st.expander("View Salary Ranges"):
            st.write("Note: Outliers are removed")
            job_api.plot_salary_ranges(data_df)

        with st.expander("Compare Your Salary to Market"):
            user_salary = st.number_input("Enter Your Salary", min_value=0)
            location = st.selectbox("Select Location (optional)", ['Nationwide'] + list(data_df['locationName'].unique()))
            if st.button("Compare Salary"):
                if location == 'Nationwide':
                    location = None
                comparison_result, user_percentile = job_api.compare_salary(user_salary, data_df, location)
                st.write(comparison_result)
        
                # Combine min and max salaries
                all_salaries = np.concatenate([data_df['minimumSalary'].dropna().values, data_df['maximumSalary'].dropna().values])
        
                # Create histogram
                fig, ax = plt.subplots()
                bin_size = 20  # Adjust as needed
                bins = np.arange(all_salaries.min(), all_salaries.max() + bin_size, bin_size)
                ax.hist(all_salaries, bins=bins, color='silver', label='Market Salary Range')
        
                # Highlight user's salary
                ax.axvline(x=user_salary, color='lightskyblue', linewidth=2, label='Your Salary')
                ax.set_xlabel('Salary')
                ax.set_ylabel('Frequency')
                ax.set_title('Your Position in the Salary Market')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
          
        with st.expander("View Jobs' details by Location"):
            unique_locations = data_df['locationName'].unique()
            selected_location = st.selectbox("Select Location to see Jobs' details", unique_locations, format_func=lambda x: '' if x is None else x)
    
            if selected_location:
                filtered_data = data_df[data_df['locationName'] == selected_location]
                       # Iterate through the filtered data and display clickable URLs
                for index, row in filtered_data.iterrows():
                    job_title = row['jobTitle']
                    salary_range = f"{row['minimumSalary']} - {row['maximumSalary']}"
                    employer = row['employerName']
                    applications = row['applications']
                    job_url = row['jobUrl']
        
                    # Create a markdown string with a clickable link
                    job_link = f"[{job_title}]({job_url})"
                    st.markdown(f"{job_link} - Salary: {salary_range} - {employer} - {applications} applications", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

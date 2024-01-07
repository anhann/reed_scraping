import pandas as pd
import numpy as np
import csv
import requests
import base64
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
from bs4 import BeautifulSoup

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import spacy
from spacy.util import filter_spans
from spacy.tokens import Span
import contractions



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

  def get_job_desc(self, data_df):
    data_df=data_df.drop_duplicates(subset=['jobDescription'])
    for i in data_df['jobId']:
      url = f'https://www.reed.co.uk/api/1.0/jobs/{i}'
      response = requests.get(url, headers=self.headers)
      data_json = response.json()
      job_description = data_json['jobDescription']
  
      soup = BeautifulSoup(job_description, 'html.parser')
      # Use newline character as separator
      job_description = soup.get_text(separator=' ')
      data_df.loc[data_df.jobId==i, 'jobDescription'] = job_description

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

    
    med_min=np.median(data_df['minimumSalary'].dropna()[data_df['minimumSalary']>1000])
    med_max=np.median(data_df['maximumSalary'].dropna()[data_df['maximumSalary']>1000])
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

    data_df = data_df[(data_df['minimumSalary'] >= 1000) & (data_df['maximumSalary'] >=1000)]

    # Remover outliers
    Q1_min = data_df['minimumSalary'].quantile(0.25)
    Q3_min = data_df['minimumSalary'].quantile(0.75)
    IQR_min = Q3_min - Q1_min
    lower_bound_min = Q1_min - 1.5 * IQR_min
    upper_bound_min = Q3_min + 1.5 * IQR_min
    data_df = data_df[(data_df['minimumSalary'] >= lower_bound_min)]

    Q1_max = data_df['maximumSalary'].quantile(0.25)
    Q3_max = data_df['maximumSalary'].quantile(0.75)
    IQR_max = Q3_max - Q1_max
    lower_bound_max = Q1_max - 1.5 * IQR_max
    upper_bound_max = Q3_max + 1.5 * IQR_max
    data_df = data_df[(data_df['maximumSalary'] <= upper_bound_max)].reset_index()


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
    plt.ylabel('Salary')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
  def compare_salary(self, user_salary, data_df, location=None):
    # Filter data by location if specified
    if location:
        data_df = data_df[data_df['locationName'] == location]
        # Remover outliers
    Q1_min = data_df['minimumSalary'].quantile(0.25)
    Q3_min = data_df['minimumSalary'].quantile(0.75)
    IQR_min = Q3_min - Q1_min
    lower_bound_min = Q1_min - 1.5 * IQR_min
    upper_bound_min = Q3_min + 1.5 * IQR_min
    data_df = data_df[(data_df['minimumSalary'] >= lower_bound_min)]

    Q1_max = data_df['maximumSalary'].quantile(0.25)
    Q3_max = data_df['maximumSalary'].quantile(0.75)
    IQR_max = Q3_max - Q1_max
    lower_bound_max = Q1_max - 1.5 * IQR_max
    upper_bound_max = Q3_max + 1.5 * IQR_max
    data_df = data_df[(data_df['maximumSalary'] <= upper_bound_max)]

    # Merge min and max salaries into a single range
    sal_df=data_df[['minimumSalary','maximumSalary']]
    sal_df=sal_df.dropna()
    all_salaries = []
    for _, row in sal_df.iterrows():
        all_salaries.extend(list(range(int(row['minimumSalary']), int(row['maximumSalary']) + 1)))

    # Calculate percentiles
    percentiles = np.percentile(all_salaries, list(range(0, 101)))
    user_percentile = np.searchsorted(percentiles, user_salary, side='right')

    # Calculate the percentage of the market the user's salary is higher than
    market_higher_than = max(0, user_percentile - 1)  # Ensures it doesn't go below 0

    comparison_result = f"Your salary is higher than {market_higher_than}% of the market."

    return comparison_result, market_higher_than



class SkillRequired:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
        self.nlp = spacy.load('en_core_web_lg')
        self.stop_words = set(stopwords.words('english'))
        self.EXP_TERMS = ['knowledge', 'skills', 'experience', 'ability', 'excellent']  # Add more terms as needed

    def clean_text(self, text):
        text = str.lower(text)
        text = re.sub(r'([.,/\\()])(?=[^\s])(?=[A-Za-z])', r'\1 ', text)
        return re.sub(r'[^\w\d\s\'-]+', '', text)

    def get_left_span(self, tok, label='', include=True):
        offset = 1 if include else 0
        idx = tok.i
        while idx > tok.left_edge.i:
            if tok.doc[idx - 1].pos_ in ('NOUN', 'ADJ', 'X'):
                idx -= 1
            else:
                break
        return label, idx, tok.i + offset

    def get_right_span(self, tok, label='', include=True):
        offset = 1 if include else 0
        idx = tok.i
        while idx < tok.right_edge.i:
            if tok.doc[idx + 1].pos_ in ('NOUN', 'ADJ', 'X'):
                idx += 1
            else:
                break
        return label, tok.i, idx + offset

    def get_conjugations(self, tok):
        new = [tok]
        while new:
            tok = new.pop()
            yield tok
            for child in tok.children:
                if child.dep_ == 'conj':
                    new.append(child)

    def extract_adp_conj_experience(self, doc, label='TERM'):
        for tok in doc:
            if tok.lower_ in self.EXP_TERMS:
                for child in tok.rights:
                    if child.dep_ == 'prep':
                        for obj in child.children:
                            if obj.dep_ == 'pobj':
                                for conj in self.get_conjugations(obj):
                                    yield self.get_left_span(conj, label)
                                    yield self.get_right_span(conj, label)

    def get_extractions(self, examples, *extractors):
        doc = self.nlp(examples, disable=['ner'])
        for ent in filter_spans([Span(doc, start, end, label) for extractor in extractors for label, start, end in extractor(doc)]):
            yield ent.text

    def list_skills(self, examples, *extractors):
        return list(self.get_extractions(examples, *extractors))


    def process_descriptions(self, df):
        # Fix contractions and clean text
        df['jobDescription'] = df['jobDescription'].apply(lambda x: ' '.join([self.clean_text(contractions.fix(word)) for word in x.split()]))
        
        # Tokenize descriptions
        df['tokenized_desc'] = df['jobDescription'].apply(word_tokenize)
        
        # Extract skills from each description
        df['skills'] = df['jobDescription'].apply(lambda x: self.list_skills(x, self.extract_adp_conj_experience))
        
        skill_list=[]
        for i in df['skills']:
          skill_list+=i

        skill_dict=dict()
        for i in skill_list:
          if i not in skill_dict.keys():
            skill_dict[i]=1
          else:
            skill_dict[i]+=1


        red= ['experience', 'skill', 'knowledge', 'role', 'ability', 'excelent', 'each', 'job', 'understanding', 'application', 'all', 'use', '', ' ']
        for word in red:
          if word in skill_dict.keys():
            del skill_dict[word]

        skill_df=pd.DataFrame(skill_dict.items(), columns=['skill','ct']).sort_values(by='ct', ascending=False)
        
        return skill_df

    def visualisation(self, df):
      skill_df = self.process_descriptions(df)

      # Select the top N skills
      top_n = 20  # You can change this number as needed
      top_skills_df = skill_df.nlargest(top_n, 'ct')

      # Plotting
      plt.figure(figsize=(12, 8))  # You can adjust the size as needed
      plt.barh(top_skills_df['skill'], top_skills_df['ct'], color='skyblue')
      plt.gca().invert_yaxis()  # Invert y-axis to have the highest count at the top

      # Add titles and labels
      plt.title('Top 20 Skills Frequency')
      plt.xlabel('Count')
      plt.ylabel('Skills')

      plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
      st.pyplot(plt)



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
        data_df=job_api.get_job_desc(data_df)
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

        with st.container(border=True):
            st.header("Number of Jobs Plot")
            selected_aggregation = st.session_state.get('aggregation', 'day')
            aggregation = st.radio("Select Aggregation Type", ('day', 'week'), index=('day', 'week').index(selected_aggregation))
            st.session_state['aggregation'] = aggregation
            job_api.plot_jobs_by_date(data_df, aggregation)

        with st.container(border=True):
            st.header("Salary Range")
            st.write("Note: Outliers are removed")
            job_api.plot_salary_ranges(data_df)

           
        with st.container(border=True):
            st.header("Compare Your Salary to Market")
            user_salary = st.number_input("Enter Your Salary", min_value=0)

            if st.button("Compare Salary"):

                location = st.selectbox("Select Location (optional)", ['Nationwide'] + list(data_df['locationName'].unique()))
                if location == 'Nationwide':
                    location = None
                comparison_result, user_percentile = job_api.compare_salary(user_salary, data_df, location)
                st.write(comparison_result)
            
                # Using Matplotlib for a stacked horizontal bar chart
                fig, ax = plt.subplots(figsize=(12,3))
                ax.barh("Salary Comparison", user_percentile+1, color='skyblue', label='Your Salary Percentile')
                ax.barh("Salary Comparison", 99 - user_percentile, left=user_percentile, color='silver', label='Rest of Market')
                ax.set_xlabel('Percentile')
                ax.set_title('Your Position in the Salary Market')
                ax.legend()
                plt.tight_layout()  # Adjust layout to fit all labels
                st.pyplot(fig)
              
        with st.container(border=True):
            st.header("Job requirements")
            skill_required = SkillRequired()
            skill_required.visualisation(data_df)

        with st.container(border=True):
            st.header("View Jobs' details by Location")
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

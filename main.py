pip install requirement.txt
import pandas as pd
import numpy as np
import csv
import requests
import base64
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from gensim import corpora, models
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class job_market:
  def __init__(self, api_key):
    self.api_key=api_key
    self.encoded_api_key = base64.b64encode(f"{api_key}:".encode()).decode()
    self.headers = {'Authorization': f'Basic {self.encoded_api_key}'}

  def get_data(self, job, location, distance):
    url = f'https://www.reed.co.uk/api/1.0/search?keywords={job}&location={location}&distancefromlocation={distance}'
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
    plt.show()
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
    plt.show()

  def extract_noun_phrases(self, text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

  def topic_modeling(self, documents):
    dictionary = corpora.Dictionary([doc.split() for doc in documents])
    corpus = [dictionary.doc2bow(doc.split()) for doc in documents]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    return lda_model.print_topics(num_words=5)

  def train_word2vec(self, documents):
    word2vec_model = Word2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model.train(documents, total_examples=len(documents), epochs=10)
    return word2vec_model

  def prepare_lstm_data(self, documents, max_length, vocab_size):
      """
      Prepares the text data for LSTM training.
      :param documents: List of text documents (job descriptions).
      :param max_length: Maximum length of the sequences.
      :param vocab_size: Size of the vocabulary.
      :return: Padded sequences of text data.
      """

      # Initialize and fit the tokenizer
      tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
      tokenizer.fit_on_texts(documents)

      # Convert text to sequences of integers
      sequences = tokenizer.texts_to_sequences(documents)

      # Pad the sequences to ensure uniform length
      padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

      return padded_sequences, tokenizer


  def train_lstm_model(self, X_train, y_train, vocab_size, embedding_dim, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    model.add(LSTM(units=50))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64)
    return model
api_key = '646bb8ba-a4bf-4d22-b895-b62fdc8a2996'

import streamlit as st

# Initialize your class (replace 'your_api_key' with your actual key)
job_api = job_market('646bb8ba-a4bf-4d22-b895-b62fdc8a2996')

# Streamlit app layout
def main():
    st.title("Job Market Analysis Tool")

    # Sidebar for input
    st.sidebar.title("Input")
    job = st.sidebar.text_input("Job Title")
    location = st.sidebar.text_input("Location")
    distance = st.sidebar.slider("Distance from Location (miles)", 0, 100, 10)
    aggregation = st.sidebar.radio("Aggregation for Job Posts by Date", ('day', 'week'))

    # Button to fetch data
    if st.sidebar.button("Fetch Data"):
        data_df = job_api.get_data(job, location, distance)
        st.write("Data Fetched Successfully")

        # Display job stats
        total_job, job_per_day, application_per_job = job_api.job_stats(data_df)
        st.write(f"Total Jobs: {total_job}, Jobs per Day: {job_per_day:.2f}, Applications per Job: {application_per_job:.2f}")

        # Plotting
        with st.expander("Show Jobs by Date"):
            job_api.plot_jobs_by_date(data_df, aggregation)

        with st.expander("Show Salary Ranges"):
            job_api.plot_salary_ranges(data_df)

# Run the app
if __name__ == "__main__":
    main()


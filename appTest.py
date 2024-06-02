import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

def sentiment_check():
    st.title("Sentiment Check")

    # NewsAPI credentials
    api_key = 'dfb4ed1de70a40428c522e22d689c882'

    # Function to fetch articles from NewsAPI
    def fetch_articles(query):
        url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
        response = requests.get(url)
        articles = response.json().get('articles', [])
        return [article['description'] for article in articles if article['description']]

    # Function to analyze sentiment
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        # Classify sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    # Fetch articles for BJP and Congress
    bjp_articles = fetch_articles('BJP 2024')
    congress_articles = fetch_articles('Congress 2024')

    # Analyze sentiment
    bjp_sentiments = [analyze_sentiment(article) for article in bjp_articles]
    congress_sentiments = [analyze_sentiment(article) for article in congress_articles]

    # Calculate sentiment percentages
    def sentiment_percentage(sentiments):
        positive = sentiments.count('positive')
        neutral = sentiments.count('neutral')
        negative = sentiments.count('negative')
        total = len(sentiments)
        return {
            'positive': positive / total * 100,
            'neutral': neutral / total * 100,
            'negative': negative / total * 100
        }

    bjp_sentiment_percentage = sentiment_percentage(bjp_sentiments)
    congress_sentiment_percentage = sentiment_percentage(congress_sentiments)

    st.write(f"BJP Sentiment: {bjp_sentiment_percentage}")
    st.write(f"Congress Sentiment: {congress_sentiment_percentage}")

    # Aggregate sentiment scores for BJP
    total_bjp_sentiment = sum(bjp_sentiment_percentage.values())

    # Aggregate sentiment scores for Congress
    total_congress_sentiment = sum(congress_sentiment_percentage.values())

    # Button for sentiment prediction
    if st.button("Check Sentiment"):
        if total_bjp_sentiment > total_congress_sentiment:
            st.write("Sentiment is leaning towards BJP.")
        elif total_bjp_sentiment < total_congress_sentiment:
            st.write("Sentiment is leaning towards Congress.")
        else:
            st.write("Sentiment is neutral towards both parties.")

def statewise_prediction():
    st.title("State-wise Prediction")
    
    # Load the data
    file_path_state = 'indianstatelevelelection.csv'
    state_data = pd.read_csv(file_path_state)
    
    # Data Wrangling
    # Drop rows with missing values in the relevant columns
    state_filtered_data = state_data.dropna(subset=['st_name', 'year', 'ac_no', 'partyabbre', 'totvotpoll'])
    
    # Determine the winning party in each constituency for each state and year
    state_winning_party_per_constituency = state_filtered_data.loc[state_filtered_data.groupby(['st_name', 'year', 'ac_no'])['totvotpoll'].idxmax()]
    
    # Filter the data to include only BJP and Congress wins
    state_bjp_congress_wins = state_winning_party_per_constituency[state_winning_party_per_constituency['partyabbre'].isin(['BJP', 'INC'])]
    
    # Count the number of constituencies won by each party for each state and year
    state_bjp_congress_wins_per_year = state_bjp_congress_wins.groupby(['st_name', 'year', 'partyabbre']).size().reset_index(name='num_wins')
    
    # Pivot the data to have separate columns for BJP and Congress wins
    state_bjp_congress_pivot = state_bjp_congress_wins_per_year.pivot(index=['st_name', 'year'], columns='partyabbre', values='num_wins').fillna(0)
    
    # Add a column for the overall winning party (1 for BJP, 0 for Congress)
    state_bjp_congress_pivot['winner'] = (state_bjp_congress_pivot['BJP'] > state_bjp_congress_pivot['INC']).astype(int)
    
    # Prepare the features (number of wins by BJP and Congress) and target variable (winner)
    X = state_bjp_congress_pivot[['BJP', 'INC']]
    y = state_bjp_congress_pivot['winner']
    
    # Time Series Analysis
    # Plot the trend in the number of constituencies won by BJP and Congress over the years
    fig1 = px.line(state_bjp_congress_pivot.groupby('year')[['BJP', 'INC']].sum(), labels={'value': 'Number of Constituencies Won', 'year': 'Year'}, title='Trend of Constituencies Won by BJP and Congress Over the Years')
    st.plotly_chart(fig1)
    
    # Naive Bayes Classification
    # Initialize the Naive Bayes model
    naive_bayes = GaussianNB()
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Naive Bayes
    naive_bayes.fit(X_train, y_train)
    y_pred_nb = naive_bayes.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    
    # K-Means Clustering
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    state_bjp_congress_pivot['cluster'] = kmeans.fit_predict(X)
    
    # Plot the clustering results
    fig2 = px.scatter(state_bjp_congress_pivot, x='BJP', y='INC', color='cluster', labels={'BJP': 'Number of Constituencies Won by BJP', 'INC': 'Number of Constituencies Won by Congress'}, title='K-Means Clustering of States Based on Election Results')
    st.plotly_chart(fig2)
    
    # Continue with the previous model training and evaluation for comparison
    # Initialize the models
    log_reg = LogisticRegression()
    random_forest = RandomForestClassifier()
    svm = SVC(kernel='linear')  # Using linear kernel to get feature importance
    
    # Train and evaluate Logistic Regression
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    
    # Train and evaluate Random Forest Classifier
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    # Train and evaluate Support Vector Machine
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    
    # Get the unique years from the pivot table
    unique_years = state_bjp_congress_pivot.index.get_level_values('year').unique()
    
    # Filter the pivot table to include only the last 5 years
    recent_years = unique_years[-5:]
    state_recent_years = state_bjp_congress_pivot[state_bjp_congress_pivot.index.get_level_values('year').isin(recent_years)]
    
    # Calculate the average wins for BJP and Congress in recent elections for each state
    state_avg_wins = state_recent_years.groupby('st_name').mean()
    
    # Function to predict the winner based on the calculated averages for each state
    def predict_winner_by_state(model, state_avg_wins):
        predictions = {}
        for state in state_avg_wins.index:
            avg_bjp_wins = state_avg_wins.loc[state, 'BJP']
            avg_inc_wins = state_avg_wins.loc[state, 'INC']
            input_data = pd.DataFrame([[avg_bjp_wins, avg_inc_wins]], columns=['BJP', 'INC'])
            prediction = model.predict(input_data)
            predictions[state] = 'BJP' if prediction[0] == 1 else 'Congress'
        return predictions
    
  # Button for Prediction
    if st.button("Predict Winner for 2024 Election"):
        # Determine the best model based on accuracy
        best_model_name = "Logistic Regression"
        best_model = log_reg
        best_accuracy = accuracy_log_reg
        
        if accuracy_rf > best_accuracy:
            best_model_name = "Random Forest"
            best_model = random_forest
            best_accuracy = accuracy_rf
        
        if accuracy_svm > best_accuracy:
            best_model_name = "SVM"
            best_model = svm
            best_accuracy = accuracy_svm
        
        if accuracy_nb > best_accuracy:
            best_model_name = "Naive Bayes"
            best_model = naive_bayes
            best_accuracy = accuracy_nb
        
        st.write(f"The best model is {best_model_name}")
        
        # Predict the winner for the 2024 election using the best model
        predicted_winner_2024 = predict_winner_by_state(best_model, state_avg_wins)
        #st.write(f"The best model is {best_model_name}")
        predictions_df = pd.DataFrame(list(predicted_winner_2024.items()), columns=['State', 'Predicted Winner'])
        # Display predictions as a table
        st.table(predictions_df)

def nationwide_prediction():
    st.title("Nationwide Prediction")
    
    # Load the data
    file_path = 'indiannationallevelelection.csv'
    data = pd.read_csv(file_path)
    
    # Filter the data to include only the relevant columns
    filtered_data = data[['year', 'pc_no', 'pc_name', 'partyabbre', 'totvotpoll']]
    
    # Determine the winning party in each constituency for each year
    winning_party_per_constituency = filtered_data.loc[filtered_data.groupby(['year', 'pc_no'])['totvotpoll'].idxmax()]
    
    # Filter the data to include only BJP and Congress wins
    bjp_congress_wins = winning_party_per_constituency[winning_party_per_constituency['partyabbre'].isin(['BJP', 'INC'])]
    
    # Count the number of constituencies won by each party for each year
    bjp_congress_wins_per_year = bjp_congress_wins.groupby(['year', 'partyabbre']).size().reset_index(name='num_wins')
    
    # Pivot the data to have separate columns for BJP and Congress wins
    bjp_congress_pivot = bjp_congress_wins_per_year.pivot(index='year', columns='partyabbre', values='num_wins').fillna(0)
    
    # Add a column for the overall winning party (1 for BJP, 0 for Congress)
    bjp_congress_pivot['winner'] = (bjp_congress_pivot['BJP'] > bjp_congress_pivot['INC']).astype(int)
    
    # Prepare the features (number of wins by BJP and Congress) and target variable (winner)
    X = bjp_congress_pivot[['BJP', 'INC']]
    y = bjp_congress_pivot['winner']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the models
    log_reg = LogisticRegression()
    random_forest = RandomForestClassifier()
    svm = SVC()
    naive_bayes = GaussianNB()
    
    # Train and evaluate Logistic Regression
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    
    # Train and evaluate Random Forest Classifier
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    # Train and evaluate Support Vector Machine
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    
    # Train and evaluate Naive Bayes
    naive_bayes.fit(X_train, y_train)
    y_pred_nb = naive_bayes.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    
    # Time Series Analysis
    # Plot the trend in the number of constituencies won by BJP and Congress over the years
    fig3 = px.line(bjp_congress_pivot, x=bjp_congress_pivot.index, y=['BJP', 'INC'],
                  labels={'value': 'Number of Constituencies Won', 'year': 'Year'},
                  title='Trend of Constituencies Won by BJP and Congress Over the Years')
    st.plotly_chart(fig3)
    
    # Clustering Analysis
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    bjp_congress_pivot['cluster'] = kmeans.fit_predict(X)
    
    # Plot the clustering results
    fig4 = px.scatter(bjp_congress_pivot, x='BJP', y='INC', color='cluster',
                     labels={'BJP': 'Number of Constituencies Won by BJP', 'INC': 'Number of Constituencies Won by Congress'},
                     title='K-Means Clustering of Election Results')
    st.plotly_chart(fig4)
    
    # Calculate the average wins for BJP and Congress in recent elections
    recent_years = bjp_congress_pivot.index[-5:]  # Use the last 5 elections for averaging
    avg_bjp_wins = bjp_congress_pivot.loc[recent_years, 'BJP'].mean()
    avg_inc_wins = bjp_congress_pivot.loc[recent_years, 'INC'].mean()
    
    st.write(f"Average BJP wins in upcoming elections prediction: {avg_bjp_wins}")
    st.write(f"Average Congress wins in upcoming elections prediction: {avg_inc_wins}")
    
    # Function to predict the winner based on the calculated averages
    def predict_winner_based_on_average(model, avg_bjp_wins, avg_inc_wins):
        input_data = pd.DataFrame([[avg_bjp_wins, avg_inc_wins]], columns=['BJP', 'INC'])
        prediction = model.predict(input_data)
        return 'BJP' if prediction[0] == 1 else 'Congress'
    
    # Button for Prediction
    if st.button("Predict Winner for 2024 Election (Nationwide)"):
        # Predict the winner for the 2024 election using each model
        predicted_winner_log_reg = predict_winner_based_on_average(log_reg, avg_bjp_wins, avg_inc_wins)
        predicted_winner_rf = predict_winner_based_on_average(random_forest, avg_bjp_wins, avg_inc_wins)
        predicted_winner_svm = predict_winner_based_on_average(svm, avg_bjp_wins, avg_inc_wins)
        predicted_winner_nb = predict_winner_based_on_average(naive_bayes, avg_bjp_wins, avg_inc_wins)

        # st.write(f"Predicted winner for 2024 election by Logistic Regression: {predicted_winner_log_reg}")
        # st.write(f"Predicted winner for 2024 election by Random Forest: {predicted_winner_rf}")
        # st.write(f"Predicted winner for 2024 election by SVM: {predicted_winner_svm}")
        # st.write(f"Predicted winner for 2024 election by Naive Bayes: {predicted_winner_nb}")

        # Determine the best model based on accuracy
        best_model_name = "Logistic Regression"
        best_model = log_reg
        best_accuracy = accuracy_log_reg

        if accuracy_rf > best_accuracy:
            best_model_name = "Random Forest"
            best_model = random_forest
            best_accuracy = accuracy_rf

        if accuracy_svm > best_accuracy:
            best_model_name = "SVM"
            best_model = svm
            best_accuracy = accuracy_svm

        if accuracy_nb > best_accuracy:
            best_model_name = "Naive Bayes"
            best_model = naive_bayes
            best_accuracy = accuracy_nb

        st.write(f"The best model is {best_model_name}")

        # Predict the winner for the 2024 election using the best model
        predicted_winner_2024 = predict_winner_based_on_average(best_model, avg_bjp_wins, avg_inc_wins)
        st.write(f"Predicted winner for 2024 election by the best model ({best_model_name}): {predicted_winner_2024}")

def main():
    st.title("Election Prediction App")
    
    tab1, tab2, tab3 = st.tabs(["State-wise Prediction", "Nationwide Prediction", "Sentiment Check"])
    
    with tab1:
        statewise_prediction()
    
    with tab2:
        nationwide_prediction()
    
    with tab3:
        sentiment_check()

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# ========== DATA COLLECTION ==========

def get_stock_data(ticker="NVDA", start_date=None, end_date=None):
    """Get stock data using yfinance"""
    stock = yf.Ticker(ticker)
    
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = stock.history(start=start_date, end=end_date)
    data = data.reset_index()
    return data

def get_news_data(query="Nvidia", days=30, api_key=None):
    """Get news data from NewsAPI"""
    if not api_key:
        # Try to load from .env file
        load_dotenv()
        api_key = os.environ.get("API_KEY")
        
    if not api_key:
        raise ValueError("API key not provided. Set API_KEY in .env file or pass as parameter.")
    
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',
        'apiKey': api_key,
        'pageSize': 100,
        'language': 'en'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data['status'] != 'ok':
        raise Exception(f"NewsAPI error: {data['message']}")
    
    articles = data['articles']
    news_data = pd.DataFrame(articles)
    news_data = news_data[['publishedAt', 'title']]
    news_data.columns = ['date', 'headline']
    
    return news_data

# ========== FEATURE ENGINEERING ==========

def preprocess_headlines(news_data):
    """Clean and preprocess news headlines"""
    try:
        stop_words = set(stopwords.words('english'))
    except:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        words = word_tokenize(text)
        words = [word for word in words if word.isalpha()]
        words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(words)
    
    news_data['clean_headlines'] = news_data['headline'].apply(preprocess_text)
    return news_data

def calculate_sentiment(news_data):
    """Calculate sentiment scores for headlines"""
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment_score(text):
        score = analyzer.polarity_scores(text)
        return score['compound']
    
    news_data['sentiment_score'] = news_data['clean_headlines'].apply(get_sentiment_score)
    return news_data

def prepare_stock_features(stock_data):
    """Prepare stock features for ML model"""
    # Create target variable - whether tomorrow's price will be higher than today's
    stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)
    
    # Create technical features - simplified for better performance on small datasets
    horizons = [2, 5, 10]  # Reduced horizon windows
    
    # Initialize new predictors list
    new_predictors = []
    
    # Numeric columns for rolling calculations
    numeric_columns = ["Close", "Volume", "Open", "High", "Low"]
    
    # Rolling average calculations
    for horizon in horizons:
        for col in numeric_columns:
            # Rolling average
            rolling_avg_column = f"{col}_RollingAvg_{horizon}"
            stock_data[rolling_avg_column] = stock_data[col].rolling(window=horizon, min_periods=1).mean()
            new_predictors.append(rolling_avg_column)
            
            # Ratio (current value / rolling average)
            ratio_column = f"{col}_Ratio_{horizon}"
            stock_data[ratio_column] = stock_data[col] / stock_data[rolling_avg_column]
            new_predictors.append(ratio_column)
    
    # Add percentage change features
    for col in numeric_columns:
        pct_change_column = f"{col}_PctChange"
        stock_data[pct_change_column] = stock_data[col].pct_change()
        new_predictors.append(pct_change_column)
    
    # Add volatility feature
    stock_data['Volatility_5day'] = stock_data['Close'].pct_change().rolling(5).std()
    new_predictors.append('Volatility_5day')
    
    # Drop any rows with NaN values
    stock_data = stock_data.dropna()
    
    return stock_data, new_predictors

def merge_stock_and_sentiment(stock_data, sentiment_data):
    """Merge stock and sentiment data on date"""
    # Convert dates to datetime format
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    
    # Find the most influential news item for each day (based on absolute sentiment score)
    def find_most_influential_news(group):
        return group.loc[abs(group['sentiment_score']).idxmax()]
        
    most_influential_news = sentiment_data.groupby('date').apply(find_most_influential_news).reset_index(drop=True)
    
    # Merge stock data with the most influential news
    combined_data = pd.merge(stock_data, most_influential_news, left_on='Date', right_on='date', how='inner')
    
    return combined_data

# ========== MODEL EVALUATION & IMPROVEMENT ==========

def analyze_feature_importance(stock_data, predictors, n_top=10):
    """Analyze feature importance in the ML model - simplified version"""
    print("\nAnalyzing feature importance...")
    
    # Prepare data
    X = stock_data[predictors]
    y = stock_data['Target']
    
    # Train model with more moderate balancing to prevent single-class predictions
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,  # Reduced for small dataset
        max_depth=10,
        class_weight={0: 1.0, 1: 1.2},  # Slight weight increase for UP class
        random_state=42
    )
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': predictors,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(n_top))
    
    try:
        # Try to create a simple plot without display
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance.head(n_top)['Feature'], feature_importance.head(n_top)['Importance'])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("Feature importance plot saved as 'feature_importance.png'")
    except Exception as e:
        print(f"Could not create plot: {str(e)}")
    
    return feature_importance.head(n_top)['Feature'].tolist()

# ========== ML MODEL ==========

def create_balanced_ml_model(stock_data, sentiment_data, predictors, test_size=5, use_selected_features=True):
    """Create a balanced ML model that addresses the UP prediction bias"""
    # Use top features if indicated
    if use_selected_features and len(predictors) > 10:
        try:
            selected_predictors = analyze_feature_importance(stock_data, predictors)
            print(f"Using selected top features: {selected_predictors}")
        except:
            print("Feature selection failed, using all features")
            selected_predictors = predictors
    else:
        selected_predictors = predictors
    
    # Prepare training and testing data
    train = stock_data.iloc[:-test_size].copy()
    test = stock_data.iloc[-test_size:].copy()
    
    # Make sure 'Date' is datetime format for both dataframes
    if 'Date' not in test.columns:
        test = test.reset_index()
        train = train.reset_index()
    
    # Ensure Date column is datetime
    test['Date'] = pd.to_datetime(test['Date']).dt.date
    
    # Print the class distribution to debug
    print(f"\nTarget distribution in training data: {train['Target'].value_counts().to_dict()}")
    
    # Initialize and train the model with moderate class weights
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,  # Reduced for small dataset
        max_depth=10,
        class_weight={0: 1.0, 1: 1.2},  # Slight weight increase for UP class
        random_state=42
    )
    
    model.fit(train[selected_predictors], train["Target"])
    
    # Get predictions
    test_predictions = model.predict(test[selected_predictors])
    
    # Get probabilities safely with error handling
    try:
        # Try to get probabilities for both classes
        test_probabilities = model.predict_proba(test[selected_predictors])
        
        # Check if we have probabilities for both classes
        if test_probabilities.shape[1] >= 2:
            test_confidence = test_probabilities[:, 1]  # Probability of upward movement
        else:
            # If only one class, use fixed confidence based on prediction
            print("Warning: Only one class in probabilities. Using fixed confidence values.")
            test_confidence = np.where(test_predictions == 1, 0.75, 0.25)
    except Exception as e:
        print(f"Error getting prediction probabilities: {str(e)}")
        print("Using fixed confidence values.")
        test_confidence = np.where(test_predictions == 1, 0.75, 0.25)
    
    # Create a DataFrame with predictions
    prediction_df = pd.DataFrame({
        'Date': test['Date'],
        'Actual_Close': test['Close'],
        'ML_Prediction': test_predictions,
        'ML_Confidence': test_confidence,
        'Actual_Target': test['Target']
    })
    
    # Merge with sentiment data
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['date']).dt.date
    combined_signals = pd.merge(
        prediction_df, 
        sentiment_data[['Date', 'sentiment_score', 'headline']], 
        on='Date', 
        how='left'
    )
    
    # Create sentiment signal (1 for positive, 0 for negative)
    combined_signals['Sentiment_Signal'] = (combined_signals['sentiment_score'] > 0).astype(int)
    
    # Create classic combined signals (ML and sentiment agree)
    combined_signals['Combined_Signal'] = np.where(
        combined_signals['ML_Prediction'] == combined_signals['Sentiment_Signal'],
        combined_signals['ML_Prediction'],
        -1  # No trade when signals disagree
    )
    
    # Create sentiment-first signals (primary signal is sentiment, ML confirms)
    combined_signals['Sentiment_First_Signal'] = np.where(
        (combined_signals['Sentiment_Signal'] == 1) & (combined_signals['ML_Confidence'] > 0.45),
        1,  # Buy when sentiment is positive and ML confidence is moderate+
        np.where(
            (combined_signals['Sentiment_Signal'] == 0) & (combined_signals['ML_Confidence'] < 0.55),
            0,  # Sell when sentiment is negative and ML confidence is moderate-
            -1  # No trade otherwise
        )
    )
    
    # Lower thresholds for Strong Signal to get more signals
    combined_signals['Strong_Signal'] = np.where(
        (combined_signals['ML_Prediction'] == 1) & 
        (combined_signals['Sentiment_Signal'] == 1) &
        (combined_signals['ML_Confidence'] > 0.52) &  # Lowered from 0.55
        (combined_signals['sentiment_score'] > 0.15),  # Lowered from 0.2
        1,  # Strong buy
        np.where(
            (combined_signals['ML_Prediction'] == 0) & 
            (combined_signals['Sentiment_Signal'] == 0) &
            (combined_signals['ML_Confidence'] > 0.52) &  # Lowered from 0.55
            (combined_signals['sentiment_score'] < -0.15),  # Changed from -0.2
            0,  # Strong sell
            -1  # No strong signal
        )
    )
    
    # Calculate next day's return
    combined_signals['Next_Day_Return'] = np.append(
        combined_signals['Actual_Close'].pct_change(1).values[1:], [np.nan]
    )
    
    # Calculate potential returns based on signals
    # For ML signal
    combined_signals['ML_Return'] = np.where(
        combined_signals['ML_Prediction'] == 1,
        combined_signals['Next_Day_Return'],
        -combined_signals['Next_Day_Return']  # For sell signals, we benefit from price drops
    )
    
    # For sentiment signal
    combined_signals['Sentiment_Return'] = np.where(
        combined_signals['Sentiment_Signal'] == 1,
        combined_signals['Next_Day_Return'],
        -combined_signals['Next_Day_Return']
    )
    
    # For combined signal
    combined_signals['Combined_Return'] = np.where(
        combined_signals['Combined_Signal'] == 1,
        combined_signals['Next_Day_Return'],
        np.where(
            combined_signals['Combined_Signal'] == 0,
            -combined_signals['Next_Day_Return'],
            0  # No trade
        )
    )
    
    # For sentiment-first signal
    combined_signals['Sentiment_First_Return'] = np.where(
        combined_signals['Sentiment_First_Signal'] == 1,
        combined_signals['Next_Day_Return'],
        np.where(
            combined_signals['Sentiment_First_Signal'] == 0,
            -combined_signals['Next_Day_Return'],
            0  # No trade
        )
    )
    
    # For strong signal
    combined_signals['Strong_Return'] = np.where(
        combined_signals['Strong_Signal'] == 1,
        combined_signals['Next_Day_Return'],
        np.where(
            combined_signals['Strong_Signal'] == 0,
            -combined_signals['Next_Day_Return'],
            0  # No trade
        )
    )
    
    return combined_signals, selected_predictors

def analyze_performance(combined_signals):
    """Analyze the performance of different trading signals"""
    # Drop the last row which will have NaN for Next_Day_Return
    df = combined_signals.dropna(subset=['Next_Day_Return']).copy()
    
    # Calculate accuracy for different signals
    ml_accuracy = accuracy_score(
        df['Actual_Target'],
        df['ML_Prediction']
    )
    
    sentiment_accuracy = accuracy_score(
        df['Actual_Target'],
        df['Sentiment_Signal']
    )
    
    # Calculate accuracy for combined signals only when they generate a trade
    combined_trades = df[df['Combined_Signal'] != -1]
    if len(combined_trades) > 0:
        combined_accuracy = accuracy_score(
            combined_trades['Actual_Target'],
            combined_trades['Combined_Signal']
        )
    else:
        combined_accuracy = np.nan
    
    # Calculate accuracy for sentiment-first signals
    sentiment_first_trades = df[df['Sentiment_First_Signal'] != -1]
    if len(sentiment_first_trades) > 0:
        sentiment_first_accuracy = accuracy_score(
            sentiment_first_trades['Actual_Target'],
            sentiment_first_trades['Sentiment_First_Signal']
        )
    else:
        sentiment_first_accuracy = np.nan
    
    # Calculate strong signal accuracy if there are any
    strong_trades = df[df['Strong_Signal'] != -1]
    if len(strong_trades) > 0:
        strong_accuracy = accuracy_score(
            strong_trades['Actual_Target'],
            strong_trades['Strong_Signal']
        )
    else:
        strong_accuracy = np.nan
    
    # Calculate average returns
    ml_avg_return = df['ML_Return'].mean()
    sentiment_avg_return = df['Sentiment_Return'].mean()
    combined_avg_return = df['Combined_Return'].replace(0, np.nan).mean()  # Ignore no-trade days
    sentiment_first_avg_return = df['Sentiment_First_Return'].replace(0, np.nan).mean()
    strong_avg_return = df['Strong_Return'].replace(0, np.nan).mean()
    
    # Calculate win rate
    ml_win_rate = (df['ML_Return'] > 0).mean()
    sentiment_win_rate = (df['Sentiment_Return'] > 0).mean()
    combined_win_rate = (df['Combined_Return'] > 0).replace(0, np.nan).mean()
    sentiment_first_win_rate = (df['Sentiment_First_Return'] > 0).replace(0, np.nan).mean()
    strong_win_rate = (df['Strong_Return'] > 0).replace(0, np.nan).mean()
    
    # Prepare results
    results = {
        'ML Accuracy': ml_accuracy,
        'Sentiment Accuracy': sentiment_accuracy,
        'Combined Accuracy': combined_accuracy,
        'Sentiment First Accuracy': sentiment_first_accuracy,
        'Strong Signal Accuracy': strong_accuracy,
        'ML Average Return': ml_avg_return,
        'Sentiment Average Return': sentiment_avg_return,
        'Combined Average Return': combined_avg_return,
        'Sentiment First Average Return': sentiment_first_avg_return,
        'Strong Signal Average Return': strong_avg_return,
        'ML Win Rate': ml_win_rate,
        'Sentiment Win Rate': sentiment_win_rate,
        'Combined Win Rate': combined_win_rate,
        'Sentiment First Win Rate': sentiment_first_win_rate,
        'Strong Signal Win Rate': strong_win_rate,
        'Trade Count': {
            'ML': len(df),
            'Sentiment': len(df),
            'Combined': (df['Combined_Signal'] != -1).sum(),
            'Sentiment First': (df['Sentiment_First_Signal'] != -1).sum(),
            'Strong': (df['Strong_Signal'] != -1).sum()
        }
    }
    
    return results

def create_text_report(combined_signals):
    """Create a detailed daily report of predictions and actual outcomes, as text only"""
    # Create prediction outcome columns
    combined_signals['ML_Outcome'] = np.where(
        combined_signals['ML_Prediction'] == combined_signals['Actual_Target'],
        'Correct',
        'Incorrect'
    )
    
    combined_signals['Sentiment_Outcome'] = np.where(
        combined_signals['Sentiment_Signal'] == combined_signals['Actual_Target'],
        'Correct',
        'Incorrect'
    )
    
    combined_signals['Sentiment_First_Outcome'] = np.where(
        (combined_signals['Sentiment_First_Signal'] == combined_signals['Actual_Target']) | 
        (combined_signals['Sentiment_First_Signal'] == -1),
        'Correct',
        'Incorrect'
    )
    
    combined_signals['Strong_Outcome'] = np.where(
        (combined_signals['Strong_Signal'] == combined_signals['Actual_Target']) | 
        (combined_signals['Strong_Signal'] == -1),
        'Correct',
        'Incorrect'
    )
    
    # Create the report data
    report_data = []
    for i, row in combined_signals.iterrows():
        # Skip rows with NaN Next_Day_Return
        if pd.isna(row['Next_Day_Return']):
            continue
            
        report = {
            'Date': row['Date'].strftime('%Y-%m-%d'),
            'Close': f"${row['Actual_Close']:.2f}",
            'ML': "UP" if row['ML_Prediction'] == 1 else "DOWN",
            'ML_Conf': f"{row['ML_Confidence']:.2f}",
            'Sentiment': "POS" if row['Sentiment_Signal'] == 1 else "NEG",
            'Sent_Score': f"{row['sentiment_score']:.2f}",
            'Sent_First': "BUY" if row['Sentiment_First_Signal'] == 1 else 
                          "SELL" if row['Sentiment_First_Signal'] == 0 else "NO TRADE",
            'Strong': "STRONG BUY" if row['Strong_Signal'] == 1 else 
                      "STRONG SELL" if row['Strong_Signal'] == 0 else "NONE",
            'Actual': "UP" if row['Actual_Target'] == 1 else "DOWN",
            'Return': f"{row['Next_Day_Return']*100:.2f}%",
            'ML_Result': row['ML_Outcome'],
            'Sent_Result': row['Sentiment_Outcome'],
            'Sent_First_Result': row['Sentiment_First_Outcome'] if row['Sentiment_First_Signal'] != -1 else "N/A",
            'Strong_Result': row['Strong_Outcome'] if row['Strong_Signal'] != -1 else "N/A"
        }
        report_data.append(report)
    
    report_df = pd.DataFrame(report_data)
    
    # Print the report in a text-friendly format
    print("\n----- DAILY SIGNALS REPORT -----")
    for _, row in report_df.iterrows():
        print(f"Date: {row['Date']} | Close: {row['Close']} | Next day: {row['Return']}")
        print(f"ML Signal: {row['ML']} ({row['ML_Conf']}) | Result: {row['ML_Result']}")
        print(f"Sentiment: {row['Sentiment']} ({row['Sent_Score']}) | Result: {row['Sent_Result']}")
        print(f"Sentiment-First: {row['Sent_First']} | Result: {row['Sent_First_Result']}")
        print(f"Strong Signal: {row['Strong']} | Result: {row['Strong_Result']}")
        print(f"Actual Move: {row['Actual']}")
        print("-" * 50)
    
    return report_df
def visualize_performance(signals):
    """Create visualizations of trading performance for all strategies"""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.dates import DateFormatter
    
    # Drop NaN values for visualization
    df = signals.dropna(subset=['Next_Day_Return']).copy()
    
    # Convert date to datetime for plotting
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 1. Price chart with signals
    ax1.plot(df['Date'], df['Actual_Close'], label='NVDA Price', color='blue', linewidth=2)
    
    # Plot ML buy signals
    ml_buy = df[df['ML_Prediction'] == 1]
    ml_sell = df[df['ML_Prediction'] == 0]
    
    ax1.scatter(ml_buy['Date'], ml_buy['Actual_Close'], marker='^', s=100, color='green', alpha=0.7, label='ML Buy')
    ax1.scatter(ml_sell['Date'], ml_sell['Actual_Close'], marker='v', s=100, color='red', alpha=0.7, label='ML Sell')
    
    # Plot combined signals with different marker
    combined_buy = df[df['Combined_Signal'] == 1]
    combined_sell = df[df['Combined_Signal'] == 0]
    if not combined_buy.empty:
        ax1.scatter(combined_buy['Date'], combined_buy['Actual_Close'], marker='*', s=200, 
                   color='lime', label='Combined Buy')
    if not combined_sell.empty:
        ax1.scatter(combined_sell['Date'], combined_sell['Actual_Close'], marker='X', s=200, 
                   color='darkred', label='Combined Sell')
    
    # Plot strong signals with different marker
    strong_buy = df[df['Strong_Signal'] == 1]
    strong_sell = df[df['Strong_Signal'] == 0]
    if not strong_buy.empty:
        ax1.scatter(strong_buy['Date'], strong_buy['Actual_Close'], marker='D', s=150, 
                   edgecolors='black', facecolors='yellow', label='Strong Buy')
    if not strong_sell.empty:
        ax1.scatter(strong_sell['Date'], strong_sell['Actual_Close'], marker='d', s=150, 
                   edgecolors='black', facecolors='orange', label='Strong Sell')
    
    # Customize first plot
    ax1.set_title('NVIDIA Stock Price with Trading Signals', fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    
    # 2. Cumulative returns comparison
    initial_investment = 1000
    
    # Calculate cumulative returns for each strategy
    df['ML_Cum_Return'] = (1 + df['ML_Return']).cumprod() * initial_investment
    df['Sentiment_Cum_Return'] = (1 + df['Sentiment_Return']).cumprod() * initial_investment
    df['Combined_Cum_Return'] = (1 + df['Combined_Return'].fillna(0)).cumprod() * initial_investment
    df['Buy_Hold_Return'] = (1 + df['Next_Day_Return']).cumprod() * initial_investment
    
    # Plot cumulative returns
    ax2.plot(df['Date'], df['ML_Cum_Return'], label='ML Strategy', color='blue', linewidth=2)
    ax2.plot(df['Date'], df['Sentiment_Cum_Return'], label='Sentiment', color='orange', linewidth=2)
    ax2.plot(df['Date'], df['Combined_Cum_Return'], label='Combined', color='green', linewidth=2)
    ax2.plot(df['Date'], df['Buy_Hold_Return'], label='Buy & Hold', color='black', linestyle='--', linewidth=2)
    
    # Customize second plot
    ax2.set_title('Cumulative Returns Comparison ($1000 Initial Investment)', fontsize=16)
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    
    # 3. Signal accuracy comparison
    strategies = ['ML', 'Sentiment', 'Combined', 'Sentiment-First', 'Strong']
    
    # Calculate accuracy for each strategy
    ml_correct = df[df['ML_Prediction'] == df['Actual_Target']].shape[0]
    ml_accuracy = ml_correct / len(df)
    
    sentiment_correct = df[df['Sentiment_Signal'] == df['Actual_Target']].shape[0]
    sentiment_accuracy = sentiment_correct / len(df)
    
    # Combined signals (excluding no-trade days)
    combined_trades = df[df['Combined_Signal'] != -1]
    if len(combined_trades) > 0:
        combined_correct = combined_trades[combined_trades['Combined_Signal'] == combined_trades['Actual_Target']].shape[0]
        combined_accuracy = combined_correct / len(combined_trades)
    else:
        combined_accuracy = 0
    
    # Sentiment-first signals
    sentfirst_trades = df[df['Sentiment_First_Signal'] != -1]
    if len(sentfirst_trades) > 0:
        sentfirst_correct = sentfirst_trades[sentfirst_trades['Sentiment_First_Signal'] == sentfirst_trades['Actual_Target']].shape[0]
        sentfirst_accuracy = sentfirst_correct / len(sentfirst_trades)
    else:
        sentfirst_accuracy = 0
    
    # Strong signals
    strong_trades = df[df['Strong_Signal'] != -1]
    if len(strong_trades) > 0:
        strong_correct = strong_trades[strong_trades['Strong_Signal'] == strong_trades['Actual_Target']].shape[0]
        strong_accuracy = strong_correct / len(strong_trades)
    else:
        strong_accuracy = 0
    
    accuracies = [ml_accuracy, sentiment_accuracy, combined_accuracy, sentfirst_accuracy, strong_accuracy]
    
    # Count trades for each strategy
    ml_trades = len(df)
    sentiment_trades = len(df)
    combined_trades = (df['Combined_Signal'] != -1).sum()
    sentfirst_trades = (df['Sentiment_First_Signal'] != -1).sum()
    strong_trades = (df['Strong_Signal'] != -1).sum()
    
    trade_counts = [ml_trades, sentiment_trades, combined_trades, sentfirst_trades, strong_trades]
    
    # Plot accuracy bars
    colors = ['blue', 'orange', 'green', 'purple', 'red']
    bar_positions = np.arange(len(strategies))
    bar_width = 0.4
    
    # Create bars
    bars = ax3.bar(bar_positions, accuracies, bar_width, color=colors, alpha=0.7)
    
    # Add trade count labels to bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{trade_counts[i]} trades', ha='center', va='bottom', fontsize=9)
        
        # Add percentage labels on bars
        ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{accuracies[i]:.1%}', ha='center', va='center', fontsize=10, 
                color='white', fontweight='bold')
    
    # Customize third plot
    ax3.set_title('Signal Accuracy Comparison', fontsize=16)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_xticks(bar_positions)
    ax3.set_xticklabels(strategies)
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add benchmark line at 50%
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax3.text(len(strategies)-1, 0.51, 'Random Guess (50%)', ha='right', va='bottom', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('nvda_performance_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance visualization saved as 'nvda_performance_visualization.png'")

# ========== MAIN EXECUTION FLOW ==========

def run_improved_analysis(api_key=None, save_data=True):
    """Run the improved analysis pipeline with balanced ML and sentiment-first approach"""
    print("Starting NVIDIA stock analysis with improved methods...")
    
    # 1. Get stock data - MODIFIED FOR 1 YEAR OF DATA
    print("Fetching stock data...")
    
    # Change the start date to get 1 year of data for ML training
    ml_start_date = "2024-03-01"  # 1 year of ML training data
    sentiment_start_date = "2025-03-01"  # 1 month of sentiment data
    
    # Get extended history for ML training
    stock_data = get_stock_data(ticker="NVDA", start_date=ml_start_date)
    
    # 2. Prepare stock features
    print("Preparing stock features...")
    stock_data, predictors = prepare_stock_features(stock_data)
    
    # 3. Get news data (still just 1 month)
    print("Fetching news data...")
    news_data = get_news_data(query="Nvidia", days=30, api_key=api_key)
    
    # 4. Process news data
    print("Processing news headlines...")
    news_data = preprocess_headlines(news_data)
    news_data = calculate_sentiment(news_data)
    
    # 5. Merge stock and sentiment data
    print("Merging stock and sentiment data...")
    combined_data = merge_stock_and_sentiment(stock_data, news_data)
    
    # Print some information about the data
    print(f"\nTotal stock data points: {len(stock_data)}")
    print(f"Stock data date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
    print(f"Sentiment data points: {len(news_data)}")
    print(f"Data points with both stock and sentiment: {len(combined_data)}")
    
    # 6. Save intermediate data if requested
    if save_data:
        stock_data.to_csv('nvda_stock_data.csv', index=False)
        news_data.to_csv('nvda_news_data.csv', index=False)
        combined_data.to_csv('nvda_combined_data.csv', index=False)
    
    # 7. Create balanced ML model with sentiment-first approach
    print("Creating balanced ML model with sentiment-first approach...")
    
    # Use the last month (points with sentiment data) as test data
    test_size = len(combined_data)
    
    # Train on longer history, test on recent month with sentiment
    signals, selected_predictors = create_balanced_ml_model(
        stock_data, combined_data, predictors, test_size, use_selected_features=True
    )
    
    # 8. Analyze performance
    print("Analyzing performance...")
    performance = analyze_performance(signals)
    
    # 9. Create daily report
    report_df = create_text_report(signals)
    
    # 10. Print summary results
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Test Period: {signals['Date'].min()} to {signals['Date'].max()}")
    print(f"Number of trading days analyzed: {len(signals)}")
    print("\nAccuracy Metrics:")
    print(f"  ML Model Accuracy: {performance['ML Accuracy']:.2%}")
    print(f"  Sentiment Analysis Accuracy: {performance['Sentiment Accuracy']:.2%}")
    if not np.isnan(performance['Combined Accuracy']):
        print(f"  Combined Signal Accuracy: {performance['Combined Accuracy']:.2%}")
    if not np.isnan(performance['Sentiment First Accuracy']):
        print(f"  Sentiment-First Accuracy: {performance['Sentiment First Accuracy']:.2%}")
    if not np.isnan(performance['Strong Signal Accuracy']):
        print(f"  Strong Signal Accuracy: {performance['Strong Signal Accuracy']:.2%}")
    
    print("\nReturn Metrics:")
    print(f"  ML Average Daily Return: {performance['ML Average Return']:.2%}")
    print(f"  Sentiment Average Daily Return: {performance['Sentiment Average Return']:.2%}")
    if not np.isnan(performance['Combined Average Return']):
        print(f"  Combined Signal Average Daily Return: {performance['Combined Average Return']:.2%}")
    if not np.isnan(performance['Sentiment First Average Return']):
        print(f"  Sentiment-First Average Daily Return: {performance['Sentiment First Average Return']:.2%}")
    if not np.isnan(performance['Strong Signal Average Return']):
        print(f"  Strong Signal Average Daily Return: {performance['Strong Signal Average Return']:.2%}")
    
    print("\nWin Rate:")
    print(f"  ML Win Rate: {performance['ML Win Rate']:.2%}")
    print(f"  Sentiment Win Rate: {performance['Sentiment Win Rate']:.2%}")
    if not np.isnan(performance['Combined Win Rate']):
        print(f"  Combined Signal Win Rate: {performance['Combined Win Rate']:.2%}")
    if not np.isnan(performance['Sentiment First Win Rate']):
        print(f"  Sentiment-First Win Rate: {performance['Sentiment First Win Rate']:.2%}")
    if not np.isnan(performance['Strong Signal Win Rate']):
        print(f"  Strong Signal Win Rate: {performance['Strong Signal Win Rate']:.2%}")
    
    print("\nNumber of Trades:")
    print(f"  ML: {performance['Trade Count']['ML']}")
    print(f"  Sentiment: {performance['Trade Count']['Sentiment']}")
    print(f"  Combined: {performance['Trade Count']['Combined']}")
    print(f"  Sentiment-First: {performance['Trade Count']['Sentiment First']}")
    print(f"  Strong: {performance['Trade Count']['Strong']}")
    
    # 11. Save daily report
    if save_data:
        report_df.to_csv('nvda_trading_signals_report.csv', index=False)
    
    visualize_performance(signals)
    
    print("\nAnalysis complete!")
    return signals, performance, report_df, combined_data, selected_predictors
# ========== EXECUTION ==========

if __name__ == "__main__":
    # Set your API key here or in .env file
    API_KEY = None  # Replace with your actual API key if not using .env
    
    # Run the analysis
    signals, performance, report_df, combined_data, selected_predictors = run_improved_analysis(api_key=API_KEY)
# NVIDIA Stock Trading Strategy Project

## **Overview**
This project implements a sophisticated trading strategy that combines *machine learning* and *news sentiment analysis* to predict price movements for NVIDIA (NVDA) stock.

## **Project Structure**
- **NvidiaRF**: Contains exploratory data analysis and initial ML model development using *RandomForestClassifier* to predict NVIDIA stock movements based on technical indicators.
- **NvidiaSentimentVolofTrade**: Focuses on sentiment analysis of news headlines related to NVIDIA, using the *VADER sentiment analyzer* to quantify the emotional tone of news.
- **Combined.py**: Integrates both approaches into a comprehensive trading system that:
  - Uses 1 year of historical price data for ML training
  - Incorporates 1 month of recent news sentiment data (historical data is paid)
  - Creates several trading signal strategies
  - Evaluates and visualizes performance

## **Core Strategies Developed**

### **1. ML Strategy**
- *Uses RandomForestClassifier* trained on technical indicators
- **Accuracy: 70.6%**
- Balanced predictions of both UP and DOWN movements

### **2. Sentiment Analysis Strategy**
- Uses *VADER* to analyze news headlines
- **Accuracy: 58.8%**
- Captures market sentiment not visible in technical data

### **3. Combined Strategy**
- Generates signals only when ML and sentiment agree
- **Accuracy: 72.7%** (highest of all)
- More selective (fewer trades)

### **4. Sentiment-First Strategy**
- Leads with sentiment and uses ML as confirmation
- **Accuracy: 61.5%**
- Alternative approach that prioritizes news sentiment

### **5. Strong Signal Strategy**
- Only triggers on high-confidence signals from both sources
- More selective (only 6 trades)
- Used for potentially larger position sizes

## **Key Achievements**
- **Improved ML Model**: Extended training data to 1 year, improving accuracy from 41% to 70.6%
- **Strategy Integration**: Successfully demonstrated that combining ML with sentiment creates more reliable signals
- **Effective Feature Selection**: Identified that *volume-related features* and *short-term price movements* are most predictive
- **Proof of Concept**: Proved approach can outperform both individual strategies and a buy-and-hold benchmark
- **Visualization**: Created insightful visualizations demonstrating performance advantages

## **Strategy Approach**
The *nvda_performance_visualization.png* graph displays:
- Stock price movements with all trading signals clearly marked
- Cumulative returns across all different strategies
- Signal accuracy comparison with trade counts for each strategy

**Note**: This project represents a robust trading system prototype that could potentially be expanded with more data sources, longer time horizons, or deployment to actual trading platforms.

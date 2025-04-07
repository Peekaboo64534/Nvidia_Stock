NVIDIA Stock Trading Strategy Project Overview
This project implements a sophisticated trading strategy that combines machine learning and news sentiment analysis to predict price movements for NVIDIA (NVDA) stock.
Project Structure

NvidiaRF: Contains exploratory data analysis and initial ML model development using RandomForestClassifier to predict NVIDIA stock movements based on technical indicators.
NvidiaSentimentVolofTrade: Focuses on sentiment analysis of news headlines related to NVIDIA, using the VADER sentiment analyzer to quantify the emotional tone of news.
Combined.py: Integrates both approaches into a comprehensive trading system that:

Uses 1 year of historical price data for ML training
Incorporates 1 month of recent news sentiment data (historical data is paid)
Creates several trading signal strategies
Evaluates and visualizes performance



Core Strategies Developed

ML Strategy: Uses RandomForestClassifier trained on technical indicators (moving averages, ratio features, volatility)

70.6% accuracy
Balanced predictions of both UP and DOWN movements


Sentiment Analysis Strategy: Uses VADER to analyze news headlines

58.8% accuracy
Captures market sentiment not visible in technical data


Combined Strategy: Generates signals only when ML and sentiment agree

72.7% accuracy (highest of all)
More selective (fewer trades)


Sentiment-First Strategy: Leads with sentiment and uses ML as confirmation

61.5% accuracy
Alternative approach that prioritizes news sentiment


Strong Signal Strategy: Only triggers on high-confidence signals from both sources

More selective (only 6 trades)
Used for potentially larger position sizes



Key Achievements

Improved ML Model: By extending training data to 1 year, improved accuracy from 41% to 70.6%
Strategy Integration: Successfully demonstrated that combining ML with sentiment creates more reliable signals (72.7% accuracy)
Effective Feature Selection: Identified that volume-related features and short-term price movements are most predictive
Proof of Concept: Despite limited sentiment data, proved that the approach can outperform both individual strategies and a buy-and-hold benchmark
Visualization: Created insightful visualizations that clearly demonstrate the performance advantages of the combined approach

Strategy Approach
The nvda_performance_visualization.png graph displays:

Stock price movements with all trading signals clearly marked
Cumulative returns across all different strategies
Signal accuracy comparison with trade counts for each strategy

This project represents a robust trading system prototype that could potentially be expanded with more data sources, longer time horizons, or deployment to actual trading platforms.

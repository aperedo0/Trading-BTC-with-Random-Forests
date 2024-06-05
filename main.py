import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("AAPL_feature_engineered_first14_removed.csv")

# Create binary target variable (1 if price increases, 0 if it decreases or stays the same)
# If next days closing price is greater than previous days closing price, price change = 1
price_change = (data['Close'].shift(-1) > data['Close']).astype(int)
# We align price change with features because we will use them for the train test split
price_change = price_change[:-1]

# Prepare features and target
# Features are the variables except for the target
features = data[['Open', 'High', 'Low', 'Volume', 'Moving_Volatility', 'RSI', 'MACD', 'Signal', 'Histogram']]
# The last day does not have a next days price which is needed for the target variable so we remove it
features = features[:-1]

# Time-based train-test split
split_idx = int(len(features) * 0.75)
X_train = features[:split_idx]
X_test = features[split_idx:]
y_train = price_change[:split_idx]
y_test = price_change[split_idx:]

# Create the random forest model using 20 decision trees
model = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=5, min_samples_leaf=5, random_state=42)
# model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

# Predict the probability of a price increase for the test set
# Predict_proba predicts the probabilities of the target variable
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate the test accuracy
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.2f}")

# Get the actual stock prices for the test set
# We only want the stock prices or y_test as used earlier
stock_prices = data['Close'][split_idx:]

# Trading strategy
def trading_strategy(probabilities, buy_threshold=0.7, sell_threshold=0.3):
    # Setting buy and sell indicies
    buy, sell = [], []
    # Last sell index
    last_sell = -1

    #loop through all propabilities
    for i, probability in enumerate(probabilities):
        if i <= last_sell:
            continue
        # If probability is greater than the buy threshold
        if probability > buy_threshold:
            buy_signal = i

            # For every value in probabilities(for the array starting at the current index, and starting j at the current index)
            for j, next_probability in enumerate(probabilities[buy_signal + 1:], start=buy_signal + 1):
                if next_probability < sell_threshold:
                    sell_signal = j
                    break
                else:
                    # If no sell signal is found, set to none because there are no sells left
                    sell_signal = None
            # If there is a sell signal append the signals and update the last sell
            if sell_signal is not None:
                buy.append(buy_signal)
                sell.append(sell_signal)
                last_sell = sell_signal
    return buy, sell

buy, sell = trading_strategy(probabilities)

# Back testing
def backtest_strategy(buy_signals, sell_signals, stock_prices, initial_capital=10000):
    capital = initial_capital
    num_shares = 0
    buy_count = 0
    sell_count = 0
    transactions = []
    profits_losses = []


    for i, stock_price in enumerate(stock_prices[:-1]):
        # If stock price index / day is in buy_signals array
        if i in buy_signals:
            num_shares_to_buy = capital // stock_price
            cost_of_total_shares = num_shares_to_buy * stock_price
            capital -= cost_of_total_shares
            num_shares += num_shares_to_buy 
            buy_count += 1
            transactions.append((stock_prices.index[i], 'Buy', stock_price))
        # If stock price index / day is in sell_signals array
        elif i in sell_signals and num_shares > 0:
            revenue_from_shares = num_shares * stock_price
            profit_or_loss = num_shares * (stock_price - stock_prices.values[buy_signals[-1]])
            capital += revenue_from_shares
            profits_losses.append(profit_or_loss)
            num_shares = 0
            sell_count += 1
            transactions.append((stock_prices.index[i], 'Sell', stock_price))
            
    final_capital = capital + num_shares * stock_prices.values[-1]
    return final_capital, buy_count, sell_count, transactions, profits_losses

initial_capital = 10000
final_capital, buy_count, sell_count, transactions, profits_losses = backtest_strategy(buy, sell, stock_prices, initial_capital=initial_capital)

# Visualize the trading strategy
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the stock prices
stock_prices_plot = data['Close'][split_idx:]
stock_prices_plot.plot(ax=ax, label='Stock Price', color='black')

# Plot buy signals
buy_dates = [stock_prices.index[i] for i in buy]
buy_prices = [stock_prices.iloc[i] for i in buy]
ax.scatter(buy_dates, buy_prices, marker='^', color='g', label='Buy')

# Plot sell signals
sell_dates = [stock_prices.index[i] for i in sell]
sell_prices = [stock_prices.iloc[i] for i in sell]
ax.scatter(sell_dates, sell_prices, marker='v', color='r', label='Sell')

# # Add labels and legend
ax.legend(loc='upper left')
ax.set_xlabel('\n03-25-2022  to  03-03-2023')
ax.set_ylabel('Stock Price')

# # Show the plot
plt.show()

# Print results and calculate percentage change from initial capital
percent_change = ((final_capital - initial_capital) / initial_capital) * 100
print(f"Final capital after the trading strategy: {final_capital} ({percent_change:.2f}%)")
print(f"Number of buy transactions: {buy_count}")
print(f"Number of sell transactions: {sell_count}")
print("Transactions (Date, Buy/Sell, Price):")
for t in transactions:
    print(t)
print("Profits and Losses for each sell transaction:")
total_profit = 0
for pl in profits_losses:
    print(f"Profit of {pl}")
    total_profit += pl
print(f"Total profit: {total_profit}")
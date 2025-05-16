import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ignore specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge.") # Specific ARIMA warning

# --- Configuration ---
DATA_FILE = 'retail_data_full_with_people_time.csv'
DIAGNOSTIC_DIR = 'diagnostic_analysis'
PREDICTIVE_DIR = 'predictive_analysis'
# Create output directories if they don't exist
os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
os.makedirs(PREDICTIVE_DIR, exist_ok=True)

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(filepath=DATA_FILE):
    """Loads and preprocesses the retail data."""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1', low_memory=False)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Convert 'InvoiceDate' to datetime
    try:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='mixed', errors='coerce')
        df.dropna(subset=['InvoiceDate'], inplace=True)
        print("Converted 'InvoiceDate' to datetime.")
    except KeyError:
        print("Error: 'InvoiceDate' column not found.")
        return None
    except Exception as e:
        print(f"Error converting 'InvoiceDate': {e}")
        return None

    # Ensure numeric types and handle errors
    numeric_cols = ['People_Count', 'Quantity', 'Price', 'Time_Spent']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found.")
    df.dropna(subset=[col for col in numeric_cols if col in df.columns], inplace=True) # Drop rows where conversion failed

    # Add derived columns
    df['Date'] = df['InvoiceDate'].dt.date
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['WeekOfYear'] = df['InvoiceDate'].dt.isocalendar().week
    df['Year'] = df['InvoiceDate'].dt.year
    df['Total_Sale'] = df['Quantity'] * df['Price']

    # Filter out returns/cancellations (negative quantity)
    df_sales = df[df['Quantity'] > 0].copy()
    df_sales.dropna(subset=['Description'], inplace=True) # Drop rows with missing descriptions

    # Convert Date back to datetime for resampling/indexing
    df_sales['Date'] = pd.to_datetime(df_sales['Date'])

    print(f"Dataset shape after cleaning: {df_sales.shape}")
    print(f"Date range: {df_sales['Date'].min()} to {df_sales['Date'].max()}")
    return df_sales

# --- Plotting Functions ---

# == Diagnostic Analysis Plots ==

def plot_product_comparison(df, product1_desc, product2_desc, filename="D1_product_comparison.png"):
    """Plots sales volume comparison for two products."""
    print(f"Generating Plot D1: Comparing '{product1_desc}' and '{product2_desc}'...")
    try:
        sales1 = df[df['Description'] == product1_desc]['Quantity'].sum()
        sales2 = df[df['Description'] == product2_desc]['Quantity'].sum()

        if pd.isna(sales1) or pd.isna(sales2):
             print(f"Warning D1: One or both products not found or have no sales data.")
             return

        plt.figure(figsize=(8, 6))
        sns.barplot(x=[product1_desc, product2_desc], y=[sales1, sales2])
        plt.title('Sales Volume Comparison')
        plt.ylabel('Total Quantity Sold')
        plt.xlabel('Product Description')
        plt.tight_layout()
        plt.savefig(os.path.join(DIAGNOSTIC_DIR, filename))
        plt.close()
        print(f"Plot D1 saved to {os.path.join(DIAGNOSTIC_DIR, filename)}")
    except Exception as e:
        print(f"Error generating Plot D1: {e}")

def plot_foot_traffic_vs_sales(df, filename="D2_foot_traffic_vs_sales.png"):
    """Plots daily foot traffic vs. daily sales count."""
    print("Generating Plot D2: Daily Foot Traffic vs. Sales Count...")
    try:
        # Aggregate daily foot traffic (sum) and sales count (unique invoices)
        daily_summary = df.groupby('Date').agg(
            Total_Foot_Traffic=('People_Count', 'sum'),
            Sales_Count=('Invoice', 'nunique') # Assuming 'Invoice' uniquely identifies a transaction
        ).reset_index()

        if daily_summary.empty:
            print("Warning D2: No data available to plot daily foot traffic vs. sales count.")
            return

        fig, ax1 = plt.subplots(figsize=(15, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Foot Traffic', color=color)
        ax1.plot(daily_summary['Date'], daily_summary['Total_Foot_Traffic'], color=color, label='Foot Traffic')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', rotation=45)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Number of Sales Transactions', color=color)
        ax2.plot(daily_summary['Date'], daily_summary['Sales_Count'], color=color, label='Sales Count')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Daily Foot Traffic vs. Daily Sales Count')
        fig.tight_layout()
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.savefig(os.path.join(DIAGNOSTIC_DIR, filename))
        plt.close(fig)
        print(f"Plot D2 saved to {os.path.join(DIAGNOSTIC_DIR, filename)}")
    except KeyError as e:
         print(f"Error generating Plot D2: Missing column - {e}")
    except Exception as e:
        print(f"Error generating Plot D2: {e}")

def plot_launch_date_vs_sales(df, product_desc, filename="D_launch_vs_sales.png", save_dir=DIAGNOSTIC_DIR):
    """Plots sales volume against the approximate launch date (first sale date) for a product."""
    print(f"Generating Plot D3/D4: Launch Date vs Sales for '{product_desc}'...")
    try:
        product_data = df[df['Description'] == product_desc].sort_values('Date')
        if product_data.empty:
            print(f"Warning D3/D4: Product '{product_desc}' not found or has no sales data.")
            return

        launch_date = product_data['Date'].min()
        total_sales = product_data['Quantity'].sum()

        # For context, maybe compare with avg sales of products launched in the same month/year?
        # Simplified: Just plot the single product's launch date and total sales.
        plt.figure(figsize=(8, 6))
        sns.barplot(x=[f"{product_desc}\n(Launched: {launch_date.strftime('%Y-%m-%d')})"], y=[total_sales])
        plt.title(f'Total Sales Volume vs. Launch Date')
        plt.ylabel('Total Quantity Sold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
        print(f"Plot D3/D4 saved to {os.path.join(save_dir, filename)}")

    except Exception as e:
        print(f"Error generating Plot D3/D4 for {product_desc}: {e}")


# == Predictive Analysis Plots ==

def get_weekly_predictions(df, stock_code, weeks_to_predict=1):
    """Trains ARIMA and predicts weekly sales for a stock code."""
    product_data = df[df['StockCode'] == stock_code].copy()
    if product_data.empty:
        print(f"Warning P1/P2/P3: No data for StockCode {stock_code}")
        return None, None

    product_data.set_index('Date', inplace=True)
    # Resample to weekly frequency, summing quantities
    weekly_sales = product_data['Quantity'].resample('W').sum()

    if len(weekly_sales) < 10: # Need sufficient data for ARIMA
        print(f"Warning P1/P2/P3: Insufficient weekly data points for StockCode {stock_code} ({len(weekly_sales)} weeks)")
        return weekly_sales, None # Return historical data even if prediction fails

    # Simple ARIMA model (can be tuned)
    try:
        # Use try-except for ARIMA fitting issues
        model = ARIMA(weekly_sales, order=(5,1,0), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=weeks_to_predict)
        # Ensure predictions are non-negative
        predictions[predictions < 0] = 0
        return weekly_sales, predictions
    except Exception as e:
        print(f"ARIMA model fitting failed for {stock_code}: {e}")
        return weekly_sales, None # Return historical data even if prediction fails


def plot_weekly_prediction(historical_sales, predictions, product_desc, stock_code, filename="P1_weekly_prediction.png"):
    """Plots historical vs. predicted weekly sales."""
    print(f"Generating Plot P1: Weekly Sales Prediction for '{product_desc}' ({stock_code})...")
    if historical_sales is None or historical_sales.empty:
         print(f"Warning P1: No historical data provided for {product_desc}.")
         return
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(historical_sales.index, historical_sales.values, label='Historical Weekly Sales')

        if predictions is not None and not predictions.empty:
            # Create future dates for predictions
            last_date = historical_sales.index[-1]
            pred_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=len(predictions), freq='W')
            plt.plot(pred_index, predictions.values, label='Predicted Weekly Sales', linestyle='--')
            # Add prediction values as text
            for i, pred in enumerate(predictions):
                plt.text(pred_index[i], pred, f'{pred:.0f}', ha='center', va='bottom')

        plt.title(f'Weekly Sales: Past vs. Predicted for {product_desc} ({stock_code})')
        plt.xlabel('Week')
        plt.ylabel('Quantity Sold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PREDICTIVE_DIR, filename))
        plt.close()
        print(f"Plot P1 saved to {os.path.join(PREDICTIVE_DIR, filename)}")
    except Exception as e:
        print(f"Error generating Plot P1 for {product_desc}: {e}")


def plot_predicted_sales_ranking(df, top_n=10, lowest=False, filename="P_predicted_ranking.png"):
    """Plots products with highest or lowest predicted sales for the next week."""
    mode = "Lowest" if lowest else "Highest"
    mode = "Lowest" if lowest else "Highest"
    print(f"Generating Plot P2/P3: Products with {mode} Predicted Sales...")
    try:
        # --- MODIFICATION START ---
        # Instead of all valid codes, predict only for a subset (e.g., top 5 most frequent)
        num_products_to_predict = 5 # Changed from 50 to 5
        recent_weeks = df['Date'].max() - pd.Timedelta(weeks=12)
        recent_sales = df[df['Date'] >= recent_weeks]
        product_counts = recent_sales['StockCode'].value_counts()

        # Get stock codes of the top N most frequently sold products recently
        top_stock_codes = product_counts.head(num_products_to_predict).index.tolist()

        if not top_stock_codes:
            print(f"Warning P2/P3: Could not find top {num_products_to_predict} products based on recent sales frequency.")
            return

        print(f"Predicting for the top {len(top_stock_codes)} most frequent products...")
        predictions_dict = {}
        processed_count = 0
        for stock_code in top_stock_codes:
            # Check if the product has enough *weekly* data points for ARIMA prediction
            hist_sales, prediction = get_weekly_predictions(df, stock_code, weeks_to_predict=1)
            # We need hist_sales to check length, even if prediction fails later
            if hist_sales is not None and len(hist_sales) >= 10 and prediction is not None and not prediction.empty:
                 # Get product description
                 desc = df[df['StockCode'] == stock_code]['Description'].iloc[0] if not df[df['StockCode'] == stock_code].empty else stock_code
                 predictions_dict[f"{desc} ({stock_code})"] = prediction.iloc[0] # Get the first week's prediction
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  Processed {processed_count}/{len(valid_stock_codes)} products...")

        if not predictions_dict:
            print("Warning P2/P3: No successful predictions were made.")
            return

        predictions_series = pd.Series(predictions_dict)
        # Filter out zero predictions for 'lowest' plot unless all are zero
        if lowest and (predictions_series > 0).any():
            predictions_series = predictions_series[predictions_series > 0]

        if predictions_series.empty:
             print(f"Warning P2/P3: No {'non-zero ' if lowest else ''}predictions available for ranking.")
             return

        ranked_predictions = predictions_series.sort_values(ascending=lowest).head(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(x=ranked_predictions.values, y=ranked_predictions.index, palette='viridis')
        plt.title(f'Top {top_n} Products with {mode} Predicted Sales Next Week')
        plt.xlabel('Predicted Quantity Sold')
        plt.ylabel('Product')
        plt.tight_layout()
        plt.savefig(os.path.join(PREDICTIVE_DIR, filename))
        plt.close()
        print(f"Plot P2/P3 saved to {os.path.join(PREDICTIVE_DIR, filename)}")

    except Exception as e:
        print(f"Error generating Plot P2/P3: {e}")


def plot_likelihood_no_sale(df, top_n=10, filename="P4_likelihood_no_sale.png"):
    """Plots products with the lowest predicted sales (proxy for likelihood of no sale)."""
    print("Generating Plot P4: Likelihood of No Sale (Proxy)...")
    # This uses the same logic as plot_predicted_sales_ranking with lowest=True
    # We are plotting the products predicted to sell the least quantity as a proxy
    plot_predicted_sales_ranking(df, top_n=top_n, lowest=True, filename=filename)


# --- Main Execution ---
if __name__ == "__main__":
    df_sales = load_and_preprocess_data()

    if df_sales is not None:
        print("\n--- Generating Diagnostic Plots ---")

        # D1: Product Comparison (Select two reasonably popular products)
        top_products = df_sales['Description'].value_counts().head(20).index.tolist()
        if len(top_products) >= 2:
            # Try to pick two distinct common items
            prod1 = top_products[0]
            prod2 = top_products[1]
            # Avoid comparing identical items if top items are variants
            if prod1.startswith("PACK OF") and prod2.startswith("PACK OF"):
                 prod2 = top_products[2] if len(top_products) > 2 else top_products[1]
            plot_product_comparison(df_sales, prod1, prod2)
        else:
            print("Warning D1: Not enough distinct products to compare.")

        # D2: Foot Traffic vs Sales
        plot_foot_traffic_vs_sales(df_sales)

        # D3 & D4: Launch Date vs Sales (Low and High Sales Products)
        product_sales_volume = df_sales.groupby('Description')['Quantity'].sum().sort_values()
        if not product_sales_volume.empty:
            low_sales_product = product_sales_volume.index[0] # Lowest total sales
            high_sales_product = product_sales_volume.index[-1] # Highest total sales
            plot_launch_date_vs_sales(df_sales, low_sales_product, filename="D3_low_sales_launch_vs_volume.png")
            plot_launch_date_vs_sales(df_sales, high_sales_product, filename="D4_high_sales_launch_vs_volume.png")
        else:
             print("Warning D3/D4: Cannot determine low/high selling products.")


        print("\n--- Generating Predictive Plots ---")

        # P1: Weekly Prediction for a specific product (e.g., a high-selling one)
        # Find StockCode for the high selling product used in D4
        high_sales_stock_code = df_sales[df_sales['Description'] == high_sales_product]['StockCode'].iloc[0] if not product_sales_volume.empty else None
        if high_sales_stock_code:
            hist_sales, preds = get_weekly_predictions(df_sales, high_sales_stock_code, weeks_to_predict=4) # Predict next 4 weeks
            if hist_sales is not None:
                plot_weekly_prediction(hist_sales, preds, high_sales_product, high_sales_stock_code, filename=f"P1_weekly_prediction_{high_sales_stock_code}.png")
        else:
            print("Warning P1: Could not find StockCode for high selling product to generate prediction plot.")


        # P2: Highest Predicted Sales
        plot_predicted_sales_ranking(df_sales, top_n=10, lowest=False, filename="P2_highest_predicted_sales.png")

        # P3: Lowest Predicted Sales
        plot_predicted_sales_ranking(df_sales, top_n=10, lowest=True, filename="P3_lowest_predicted_sales.png")

        # P4: Likelihood of No Sale (Proxy using lowest predicted)
        plot_likelihood_no_sale(df_sales, top_n=10, filename="P4_likelihood_no_sale.png")

        print("\n--- Plot Generation Complete ---")
    else:
        print("\n--- Plot Generation Failed: Could not load data ---")

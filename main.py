import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

def load_data(filename):
    """
    Load and prepare sales data from CSV file
    """
    # Load the data
    df = pd.read_csv(filename)
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Handle negative quantities (returns) - either remove or keep based on business logic
    # df = df[df['Quantity'] > 0]  # Uncomment if you want to exclude returns
    
    return df

class StockPredictor:
    def __init__(self, data):
        """Initialize with sales data"""
        self.data = data
        self.models = {}  # Dictionary to store models for each product
    
    def aggregate_daily_sales(self, product_id, days_history=90):
        """
        Create daily time series for a specific product
        Returns a Series with date index and daily sales quantities
        """
        # Filter for the specific product
        product_data = self.data[self.data['StockCode'] == product_id].copy()
        
        if product_data.empty:
            print(f"No data found for product {product_id}")
            return None
        
        # Get the date range we need
        max_date = product_data['InvoiceDate'].max()
        min_date = max_date - pd.Timedelta(days=days_history)
        
        # Filter to only the period we need (reduces memory usage)
        product_data = product_data[product_data['InvoiceDate'] >= min_date]
        
        # Aggregate by day
        daily_sales = product_data.groupby(
            pd.Grouper(key='InvoiceDate', freq='D')
        )['Quantity'].sum()
        
        # Create full date range and fill missing values with 0
        full_range = pd.date_range(start=min_date, end=max_date, freq='D')
        daily_sales = daily_sales.reindex(full_range, fill_value=0)
        
        return daily_sales
    
    def train_model(self, product_id):
        """
        Train an ARIMA model for a specific product
        """
        # Get daily time series
        daily_sales = self.aggregate_daily_sales(product_id)
        
        if daily_sales is None or len(daily_sales) < 14:  # Need at least 2 weeks of data
            print(f"Insufficient data for product {product_id}")
            return None
        
        print(f"Training model for product {product_id}")
        
        try:
            # Use auto_arima for automatic parameter selection
            # Simplified parameters for faster execution
            model = auto_arima(
                daily_sales,
                seasonal=True,    # Enable seasonality
                m=7,              # Weekly seasonality
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                d=1,              # Usually first differencing works well
                D=1,              # Seasonal differencing
                stepwise=True,    # Faster parameter search
                suppress_warnings=True,
                error_action='ignore',
                max_order=10,
                maxiter=50
            )
            
            self.models[product_id] = model
            return model
            
        except Exception as e:
            print(f"Error training model for product {product_id}: {str(e)}")
            return None
    
    def get_product_info(self, product_id):
        """Get basic information about a product"""
        product_data = self.data[self.data['StockCode'] == product_id]
        
        if product_data.empty:
            return None
        
        return {
            'description': product_data['Description'].iloc[0].strip(),
            'total_sales': product_data['Quantity'].sum(),
            'avg_daily_sales': self.aggregate_daily_sales(product_id).mean() if self.aggregate_daily_sales(product_id) is not None else 0
        }
    
    def predict_stock_needs(self, product_id, days=14):
        """
        Predict stock needs for the next specified number of days
        """
        # Train model if not already trained
        if product_id not in self.models:
            self.train_model(product_id)
        
        if product_id not in self.models:
            print(f"Could not train model for product {product_id}")
            return self._fallback_prediction(product_id, days)
        
        try:
            # Get product info
            product_info = self.get_product_info(product_id)
            
            # Make forecast
            forecast = self.models[product_id].predict(n_periods=days)
            
            # Ensure predictions are non-negative and reasonable
            forecast = np.maximum(forecast, 0)
            
            # Round to integers (can't have fractional inventory)
            forecast = np.round(forecast).astype(int)
            
            # Create date range for forecast - FIXED to ensure datetime type
            last_date = self.data['InvoiceDate'].max().date()
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            # Create forecast DataFrame with explicit datetime conversion
            forecast_df = pd.DataFrame({
                'date': pd.to_datetime(forecast_dates),  # Explicitly convert to pandas datetime
                'predicted_quantity': forecast
            })
            
            return {
                'product_id': product_id,
                'description': product_info['description'],
                'daily_forecast': forecast_df,
                'total_stock_needed': int(forecast.sum()),
                'avg_daily_need': float(forecast.mean())
            }
            
        except Exception as e:
            print(f"Error predicting stock for product {product_id}: {str(e)}")
            return self._fallback_prediction(product_id, days)

    def _fallback_prediction(self, product_id, days):
        """Simple fallback if model fails"""
        product_info = self.get_product_info(product_id)
        if product_info is None:
            return None
        
        # Use recent average as fallback
        daily_sales = self.aggregate_daily_sales(product_id, days_history=30)
        if daily_sales is None:
            avg_daily = 1  # Minimum baseline
        else:
            avg_daily = max(1, daily_sales.mean())
        
        # Create fallback forecast - FIXED to ensure datetime type
        last_date = self.data['InvoiceDate'].max().date()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        forecast_qty = [round(avg_daily)] * days
        
        return {
            'product_id': product_id,
            'description': product_info['description'],
            'daily_forecast': pd.DataFrame({
                'date': pd.to_datetime(forecast_dates),  # Explicitly convert to pandas datetime
                'predicted_quantity': forecast_qty
            }),
            'total_stock_needed': int(sum(forecast_qty)),
            'avg_daily_need': float(avg_daily),
            'note': 'Using fallback prediction based on historical average'
        }
    
    def visualize_forecast(self, product_id, days=14):
        """
        Visualize historical sales and forecast
        """
        # Get historical data
        daily_sales = self.aggregate_daily_sales(product_id)
        if daily_sales is None:
            print(f"No data to visualize for product {product_id}")
            return
        
        # Get forecast
        forecast_result = self.predict_stock_needs(product_id, days)
        if forecast_result is None:
            print(f"Could not generate forecast for product {product_id}")
            return
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data (last 30 days)
        recent_history = daily_sales[-30:]
        plt.plot(recent_history.index, recent_history.values, 
                label='Historical Sales', color='blue')
        
        # Plot forecast
        forecast_df = forecast_result['daily_forecast']
        plt.plot(forecast_df['date'], forecast_df['predicted_quantity'],
                label='Predicted Stock Needed', color='red', linestyle='--')
        
        # Add vertical line for current time
        plt.axvline(x=daily_sales.index[-1], color='green', linestyle='-', 
                   alpha=0.5, label='Current Date')
        
        # Formatting
        plt.title(f"Stock Prediction - {forecast_result['description']}")
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt

def main():
    """
    Main function to run the stock prediction system
    """
    filename = input("Enter the path to your CSV file: ")
    
    # Load data
    print("Loading data...")
    try:
        data = load_data(filename)
        print(f"Loaded {len(data)} records.")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Create predictor
    predictor = StockPredictor(data)
    
    while True:
        print("\n===== Stock Prediction System =====")
        print("1. Predict stock needs for a product")
        print("2. View product information")
        print("3. List top selling products")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            product_id = input("Enter product StockCode: ")
            days = int(input("Enter number of days to predict (default 14): ") or 14)
            
            # Check if product exists
            if predictor.get_product_info(product_id) is None:
                print(f"Product {product_id} not found in data")
                continue
            
            # Generate prediction
            result = predictor.predict_stock_needs(product_id, days)
            
            if result:
                print(f"\nProduct: {result['description']}")
                print(f"Total stock needed for next {days} days: {result['total_stock_needed']} units")
                print(f"Average daily need: {result['avg_daily_need']:.2f} units")
                
                # Show daily breakdown
                print("\nDaily stock needs:")
                forecast_table = result['daily_forecast'].copy()
                forecast_table['date'] = forecast_table['date'].dt.strftime('%Y-%m-%d')
                print(forecast_table.to_string(index=False))
                
                # Visualize
                predictor.visualize_forecast(product_id, days)
                plt.show()
            
        elif choice == '2':
            product_id = input("Enter product StockCode: ")
            info = predictor.get_product_info(product_id)
            
            if info:
                print(f"\nProduct: {info['description']}")
                print(f"Total historical sales: {info['total_sales']} units")
                print(f"Average daily sales: {info['avg_daily_sales']:.2f} units")
            else:
                print(f"Product {product_id} not found in data")
                
        elif choice == '3':
            # Show top selling products
            top_n = int(input("How many top products to show? (default 10): ") or 10)
            
            product_sales = data.groupby(['StockCode', 'Description'])['Quantity'].sum().reset_index()
            product_sales = product_sales.sort_values('Quantity', ascending=False).head(top_n)
            
            print(f"\nTop {top_n} Selling Products:")
            for i, (_, row) in enumerate(product_sales.iterrows(), 1):
                print(f"{i}. {row['StockCode']} - {row['Description'].strip()}: {row['Quantity']} units")
                
        elif choice == '4':
            print("Exiting. Thank you for using the Stock Prediction System!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()

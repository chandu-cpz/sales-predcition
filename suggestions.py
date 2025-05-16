import pandas as pd
# import ollama # Remove ollama import
import google.generativeai as genai # Add google
import sys
import os # Import os module

# --- Configuration ---
# Construct the absolute path to the dataset relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "retail_data_full_with_people_time.csv")
# !! IMPORTANT: Set your Google API Key as an environment variable !!
# Example: export GOOGLE_API_KEY='YOUR_API_KEY'
# Or add it directly here (less secure): genai.configure(api_key="YOUR_API_KEY")
GOOGLE_API_KEY = "AIzaSyAlV6jPPwaj669vdEigWnfyLBrel4xKWns" 
GOOGLE_MODEL_NAME = "gemini-1.5-flash-latest" # Or another suitable model like "gemini-pro"
# --- ---

# Configure the Generative AI library
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
    print("Please set the GOOGLE_API_KEY environment variable before running the script.", file=sys.stderr)
    sys.exit(1)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Safety settings can be adjusted here if needed
    # generation_config = genai.types.GenerationConfig(temperature=0.7) 
    model = genai.GenerativeModel(GOOGLE_MODEL_NAME) 
except Exception as e:
    print(f"Error configuring Google Generative AI: {e}", file=sys.stderr)
    sys.exit(1)

def get_average_sales(stock_code: str) -> float:
    """
    Calculates the average sales quantity for a given stock code from the dataset.

    Args:
        stock_code: The product stock code.

    Returns:
        The average sales quantity, or 0.0 if the stock code is not found
        or if there's an error reading the data.
    """
    try:
        df = pd.read_csv(DATASET_PATH, encoding='utf-8', on_bad_lines='skip')
        # Ensure StockCode is treated as string for accurate matching
        df['StockCode'] = df['StockCode'].astype(str)

        # Convert Quantity to numeric, coercing errors to NaN, then filter
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df.dropna(subset=['Quantity'], inplace=True) # Remove rows where Quantity became NaN
        df['Quantity'] = df['Quantity'].astype(int) # Convert valid quantities to int

        product_sales = df[df['StockCode'] == stock_code]['Quantity']

        if product_sales.empty:
            print(f"Warning: StockCode '{stock_code}' not found or had no valid numeric quantity in the dataset.", file=sys.stderr)
            return 0.0
        
        # Consider only positive quantities for average calculation? 
        # Depending on data meaning, negative might mean returns.
        # For now, let's include all.
        average = product_sales.mean()
        return average if pd.notna(average) else 0.0

    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{DATASET_PATH}'", file=sys.stderr)
        return 0.0
    except Exception as e:
        print(f"Error processing dataset: {e}", file=sys.stderr)
        return 0.0

def generate_suggestion(stock_code: str, current_sales: int, average_sales: float) -> str:
    """
    Generates sales suggestions using the Google Generative AI model based on sales performance.

    Args:
        stock_code: The product stock code.
        current_sales: The current sales figure provided by the user.
        average_sales: The calculated average sales for this product.

    Returns:
        A suggestion string from the Google Generative AI model.
    """
    if average_sales == 0.0: # Handle cases where average couldn't be calculated
         prompt = f"The product with StockCode '{stock_code}' was not found in historical sales data or there was an error. The current reported sales figure is {current_sales}. Provide general advice for a product with this sales number, considering it might be a new or problematic item."
    elif current_sales < average_sales:
        prompt = f"Sales for product StockCode '{stock_code}' are currently {current_sales}, which is below the average of {average_sales:.2f}. Suggest specific actions or strategies to increase sales for this product."
    else:
        prompt = f"Sales for product StockCode '{stock_code}' are currently {current_sales}, which is at or above the average of {average_sales:.2f}. Suggest actions related to inventory management, like restocking, and potentially capitalizing on its popularity."

    print(f"\n--- Prompting {GOOGLE_MODEL_NAME} ---") # Use Google model name
    print(prompt)
    print("---------------------------\n")

    try:
        # Use the configured Google AI model
        response = model.generate_content(prompt) 
        # Add basic error handling for response structure if needed
        if response.parts:
             return response.text
        else:
             # Handle cases where the response might be blocked or empty
             return f"Received an empty or blocked response from {GOOGLE_MODEL_NAME}. Reason: {response.prompt_feedback}"
            
    except Exception as e:
        # Update error message for Google AI
        return f"Error communicating with Google Generative AI model '{GOOGLE_MODEL_NAME}': {e}"

if __name__ == "__main__":
    print("Sales Suggestion Tool")
    print("---------------------")

    while True:
        try:
            input_stock_code = input("Enter the Product Stock Code (e.g., 85048): ").strip()
            if not input_stock_code:
                print("Stock Code cannot be empty.")
                continue
            break
        except EOFError:
            print("\nExiting.")
            sys.exit(0)


    while True:
        try:
            input_sales_str = input(f"Enter the current sales quantity for {input_stock_code}: ").strip()
            input_sales = int(input_sales_str)
            break
        except ValueError:
            print("Invalid input. Please enter a whole number for sales quantity.")
        except EOFError:
            print("\nExiting.")
            sys.exit(0)

    print(f"\nCalculating average sales for StockCode: {input_stock_code}...")
    avg_sales = get_average_sales(input_stock_code)

    if avg_sales is not None:
        print(f"Average sales quantity: {avg_sales:.2f}")
        print("Generating suggestion...")
        suggestion = generate_suggestion(input_stock_code, input_sales, avg_sales)
        print("\n--- Suggestion ---")
        print(suggestion)
        print("------------------")

    else:
        print("Could not generate suggestion due to previous errors.")

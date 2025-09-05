import pandas as pd
from .training_model import LR, DT, GB, RF, vectorization, wordopt  # Import models and functions from training_model.py

def output_label(n):
    """Convert numerical label to string label."""
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    else:
        return "Unknown Label"

def manual_testing(news):
    """Process the news article and make predictions using trained models."""
    # Create a DataFrame for the input news
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    
    # Preprocess the text
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    
    # Transform the text using the vectorizer
    new_xv_test = vectorization.transform(new_x_test)
    
    # Make predictions using the trained models
    try:
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GB = GB.predict(new_xv_test)
        pred_RF = RF.predict(new_xv_test)

        # Print predictions
        print("\n\nLR Prediction: {}  \nDT Prediction: {} \nGB Prediction: {} \nRF Prediction: {}".format(
            output_label(pred_LR[0]), 
            output_label(pred_DT[0]), 
            output_label(pred_GB[0]), 
            output_label(pred_RF[0])
        ))
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

# Example usage
if __name__ == "__main__":
    news_input = input("Enter the news article text for testing: ").strip()
    if news_input:
        manual_testing(news_input)
    else:
        print("No input provided. Please enter a news article to test.")
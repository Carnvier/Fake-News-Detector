from django.shortcuts import render
from django.http import JsonResponse
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from .manual_testing import manual_testing, output_label
from .training_model import LR, vectorization, wordopt

class IndexView(TemplateView):
    template_name = "index.html"  # Specify your template name here

@csrf_exempt  # Allow CSRF exemption for this view
def predict_view(request):
    if request.method == 'POST':
        try:
            news_text = request.POST.get('news_text', '').strip()
            if not news_text:
                return JsonResponse({'error': 'No text provided'}, status=400)

            # Create a DataFrame for the input text
            testing_news = {"text": [news_text]}
            new_def_test = pd.DataFrame(testing_news)

            # Apply preprocessing (word optimization)
            new_def_test["text"] = new_def_test["text"].apply(wordopt)
            new_x_test = new_def_test["text"]

            # Transform the text into the model's expected format
            new_xv_test = vectorization.transform(new_x_test)

            # Make prediction using Logistic Regression (primary model)
            pred_LR = LR.predict(new_xv_test)
            confidence = LR.predict_proba(new_xv_test).max()

            response_data = {
                'prediction': int(pred_LR[0]),  # Convert prediction to integer
                'confidence': float(confidence),  # Convert confidence to float
                'label': output_label(pred_LR[0])  # Get the label for the prediction
            }

            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)
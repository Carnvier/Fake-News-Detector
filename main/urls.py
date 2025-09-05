from django.urls import path
from .views import IndexView, predict_view

urlpatterns = [
    path('', IndexView.as_view(), name='index'),  # Root URL for rendering the index page
    path('predict/', predict_view, name='predict'),  # URL for making predictions
]
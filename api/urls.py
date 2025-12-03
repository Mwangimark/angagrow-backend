from django.urls import path
from .views import (
    ChatbotView,
    CropAnalysisView,
)

urlpatterns = [
    path("crop-analysis/", CropAnalysisView.as_view(), name="crop-analysis"),
    path("chatbot/", ChatbotView.as_view(), name="chatbot"),
]

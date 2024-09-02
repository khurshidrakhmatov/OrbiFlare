from django.urls import path
from .views import NewsCreateView, NewsMakerAverageScoreView

urlpatterns = [
    # URL pattern for creating a new news item
    path('news/', NewsCreateView.as_view(), name='news-create'),
    path('news_makers/', NewsMakerAverageScoreView.as_view(), name='news-averages-list'),
]

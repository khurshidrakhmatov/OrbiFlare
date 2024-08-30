from django.urls import path
from .views import NewsCreateView, NewsAverageView

urlpatterns = [
    # URL pattern for creating a new news item
    path('news/', NewsCreateView.as_view(), name='news-create'),

    # URL pattern for getting average reliability scores
    path('news/averages/', NewsAverageView.as_view(), name='news-averages'),
]

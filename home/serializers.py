from rest_framework import serializers
from .models import NewsModel


class NewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsModel
        fields = ['news_maker', 'news_text', 'reliability_score']
        read_only_fields = ['reliability_score']

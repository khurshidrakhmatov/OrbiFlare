# home/views.py
from rest_framework import generics
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ValidationError
from .models import NewsModel
from .serializers import NewsSerializer
from .trained_model import main_func, source_information, fake_news, quran_verses
from django.db.models import Avg


class NewsCreateView(generics.CreateAPIView):
    queryset = NewsModel.objects.all()
    serializer_class = NewsSerializer

    def post(self, request, *args, **kwargs):
        news_text = request.data.get('news_text', None)
        news_maker = request.data.get('news_maker', None)

        if not news_text or not news_maker:
            return Response({
                'error': 'Both "news_text" and "news_maker" fields are required.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Calculate reliability score
        try:
            reliability_score = main_func(news_text, source_information, fake_news, quran_verses)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Convert reliability score to integer
        reliability_score = int(reliability_score)

        # Create and save the news item
        news_item = NewsModel(news_maker=news_maker, news_text=news_text, reliability_score=reliability_score)
        news_item.save()

        # Serialize the new news item and return the response
        serializer = self.get_serializer(news_item)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class NewsAverageView(generics.ListAPIView):
    serializer_class = NewsSerializer

    def get(self, request, *args, **kwargs):
        news_maker_avg = (
            NewsModel.objects
            .values('news_maker')
            .annotate(average_reliability=Avg('reliability_score'))
        )
        result = [
            {'news_maker': item['news_maker'], 'average_reliability': int(item['average_reliability'])}
            for item in news_maker_avg
        ]
        return Response(result)

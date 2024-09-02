from rest_framework.response import Response
from .models import NewsModel
from .serializers import NewsSerializer, NewsMakerAverageScoreSerializer
from .trained_model import main_func  # Import your trained model function
import pickle
from django.db.models import Avg
from rest_framework import generics, response
from rest_framework.views import APIView


class NewsCreateView(generics.ListCreateAPIView):
    queryset = NewsModel.objects.all()
    serializer_class = NewsSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        news_maker = serializer.validated_data.get('news_maker', '').capitalize()
        news_text = serializer.validated_data.get('news_text')

        # Load datasets (you may want to load these datasets once and cache them)
        with open('source_information_processed.pkl', 'rb') as f:
            source_information = pickle.load(f)

        with open('fake_news_processed.pkl', 'rb') as f:
            fake_news = pickle.load(f)

        with open('Quran_verses.pkl', 'rb') as f:
            quran_verses = pickle.load(f)

        # Compute the reliability score
        overall_score = main_func(news_text, source_information, fake_news, quran_verses)
        # overall_score = len(news_text)

        # Save the instance with the computed reliability_score
        instance = serializer.save(news_maker=news_maker, reliability_score=overall_score)

        # Return only the reliability_score in the response
        return response.Response({"reliability_score": instance.reliability_score})


class NewsMakerAverageScoreView(APIView):
    def get(self, request):
        # Aggregate average reliability score by news_maker
        average_scores = NewsModel.objects.values('news_maker').annotate(
            average_reliability_score=Avg('reliability_score')
        ).order_by('news_maker')

        # Serialize the results
        serializer = NewsMakerAverageScoreSerializer(average_scores, many=True)
        return response.Response(serializer.data)

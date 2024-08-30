from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from .models import NewsModel


class NewsAPITestCase(APITestCase):

    def setUp(self):
        self.create_url = reverse('news-create')
        self.averages_url = reverse('news-averages')

        # Create test data programmatically if needed
        NewsModel.objects.create(news_text='Sample news text 1', news_maker='Sample Maker 1', reliability_score=75)
        NewsModel.objects.create(news_text='Sample news text 2', news_maker='Sample Maker 1', reliability_score=85)
        NewsModel.objects.create(news_text='Sample news text 3', news_maker='Sample Maker 2', reliability_score=65)

    def test_create_news_valid(self):
        data = {
            'news_text': 'This is a sample news text.',
            'news_maker': 'Sample Maker'
        }
        response = self.client.post(self.create_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('news_text', response.data)
        self.assertIn('news_maker', response.data)
        self.assertIn('reliability_score', response.data)

    def test_create_news_missing_fields(self):
        data = {
            'news_text': 'This news text is missing the maker field.'
        }
        response = self.client.post(self.create_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data['error'], 'Both "news_text" and "news_maker" fields are required.')

    def test_get_news_averages(self):
        response = self.client.get(self.averages_url, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)
        if response.data:
            self.assertIn('news_maker', response.data[0])
            self.assertIn('average_reliability', response.data[0])

    def test_get_news_averages_no_data(self):
        # Delete all news items to test the no data case
        NewsModel.objects.all().delete()
        response = self.client.get(self.averages_url, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, [])

    def test_create_news_invalid_data(self):
        data = {
            'news_text': 'Some text without a news maker'
        }
        response = self.client.post(self.create_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data['error'], 'Both "news_text" and "news_maker" fields are required.')

    @classmethod
    def tearDownClass(cls):
        # Perform cleanup tasks if needed
        NewsModel.objects.all().delete()
        super().tearDownClass()

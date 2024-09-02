from django.db import models


# Create your models here.

class NewsModel(models.Model):
    news_maker = models.CharField(max_length=255)
    news_text = models.TextField()
    reliability_score = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.news_maker

from django.db import models

class County(models.Model):
    county_name = models.CharField(max_length=100, unique=True)
    total_population = models.IntegerField()
    total_uninsured_population = models.IntegerField()
    area_land_sqmi = models.FloatField()
    area_water_sqmi = models.FloatField()
    state = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.county_name}, {self.state}"

class Hospital(models.Model):
    provider_id = models.CharField(max_length=50, unique=True)
    hospital_name = models.CharField(max_length=200)
    rating = models.FloatField()
    county = models.ForeignKey(County, on_delete=models.CASCADE, related_name='hospitals', default=None)  # Add temporary default
    state = models.CharField(max_length=100)

    def __str__(self):
        return self.hospital_name

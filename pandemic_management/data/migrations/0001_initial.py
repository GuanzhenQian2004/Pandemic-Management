# Generated by Django 5.1.5 on 2025-01-26 03:11

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='County',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('county_name', models.CharField(max_length=100, unique=True)),
                ('total_population', models.IntegerField()),
                ('total_uninsured_population', models.IntegerField()),
                ('area_land_sqmi', models.FloatField()),
                ('area_water_sqmi', models.FloatField()),
                ('state', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Hospital',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('provider_id', models.CharField(max_length=50, unique=True)),
                ('hospital_name', models.CharField(max_length=200)),
                ('rating', models.FloatField()),
                ('state', models.CharField(max_length=100)),
                ('county', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='hospitals', to='data.county')),
            ],
        ),
    ]

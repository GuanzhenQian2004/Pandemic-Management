# Generated by Django 5.1.5 on 2025-01-26 04:47

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0002_remove_hospital_county_hospital_counties'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='hospital',
            name='counties',
        ),
        migrations.AddField(
            model_name='hospital',
            name='county',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, related_name='hospitals', to='data.county'),
        ),
    ]

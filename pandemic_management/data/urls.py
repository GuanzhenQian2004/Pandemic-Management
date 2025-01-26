from django.urls import path
from .views import upload_csv, view_data, wipe_data, error_page

urlpatterns = [
    path('upload_csv/', upload_csv, name='upload_csv'),
    path('view_data/', view_data, name='view_data'),
    path('wipe_data/', wipe_data, name='wipe_data'),
    path('error_page/', error_page, name='error_page'),
]
import csv
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import County, Hospital
from .forms import CSVUploadForm

def upload_csv(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            county_file = request.FILES['county_file']
            hospital_file = request.FILES['hospital_file']
            try:
                # Process County CSV
                county_reader = csv.DictReader(county_file.read().decode('utf-8').splitlines())
                for row in county_reader:
                    _, created = County.objects.get_or_create(
                        county_name=row['county_name'],
                        defaults={
                            'total_population': int(row['total_population']),
                            'total_uninsured_population': int(row['total_uninsured_population']),
                            'area_land_sqmi': float(row['area_land_sqmi']),
                            'area_water_sqmi': float(row['area_water_sqmi']),
                            'state': row['state']
                        }
                    )
                    if not created:
                        messages.warning(request, f"County '{row['county_name']}' already exists.")

                # Process Hospital CSV
                hospital_reader = csv.DictReader(hospital_file.read().decode('utf-8').splitlines())
                for row in hospital_reader:
                    try:
                        county = County.objects.get(county_name=row['county_name'])
                        _, created = Hospital.objects.get_or_create(
                            provider_id=row['provider_id'],
                            defaults={
                                'hospital_name': row['hospital_name'],
                                'rating': float(row['rating']),
                                'county': county,
                                'state': row['state']
                            }
                        )
                        if not created:
                            messages.warning(request, f"Hospital '{row['hospital_name']}' already exists.")
                    except County.DoesNotExist:
                        messages.error(request, f"Error: County '{row['county_name']}' does not exist.")
                
                return redirect('view_data')  # Redirect to view current data on success
            except Exception as e:
                messages.error(request, f"Error processing CSV files: {e}")
                return redirect('error_page')  # Redirect to error page
    else:
        form = CSVUploadForm()
    return render(request, 'data/upload_csv.html', {'form': form})

def view_data(request):
    counties = County.objects.all()
    hospitals = Hospital.objects.all()
    return render(request, 'data/view_data.html', {'counties': counties, 'hospitals': hospitals})

def wipe_data(request):
    try:
        Hospital.objects.all().delete()
        County.objects.all().delete()
        messages.success(request, "All data has been wiped successfully!")
        return redirect('view_data')  # Redirect to view current data
    except Exception as e:
        messages.error(request, f"Error wiping data: {e}")
        return redirect('error_page')  # Redirect to error page

def error_page(request):
    return render(request, 'data/error_page.html')

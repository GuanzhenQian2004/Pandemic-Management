from django import forms

class CSVUploadForm(forms.Form):
    county_file = forms.FileField(label="County CSV File")
    hospital_file = forms.FileField(label="Hospital CSV File")

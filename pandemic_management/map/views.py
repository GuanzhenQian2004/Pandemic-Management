import os
import json
import logging
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

logger = logging.getLogger(__name__)

# JSON View to load data
def json_view(request):
    # Dynamically construct the file path
    file_path = os.path.join(settings.BASE_DIR, 'map', 'static', 'data', 'combined_data.json')
    
    try:
        # Try reading the file with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding if utf-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {str(e)}")
            return JsonResponse({'error': 'Invalid file encoding'}, status=500)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return JsonResponse({'error': 'File not found'}, status=404)
    except Exception as e:
        logger.error(f"Unexpected error loading JSON: {str(e)}")
        return JsonResponse({'error': 'Unexpected error'}, status=500)
    
    # Return the loaded JSON data
    return JsonResponse(data, safe=False)

# View to render the map template
def county_map(request):
    # Render the map.html template
    return render(request, 'map.html')

from .models import DroneImage

def get_latest_analysis():
    try:
        return DroneImage.objects.latest('timestamp')
    except DroneImage.DoesNotExist:
        return None

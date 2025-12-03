from rest_framework import serializers
from .models import DroneImage

class DroneImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = DroneImage
        fields = "__all__"
        read_only_fields = [
            "vari", "gli", "exg", 
            "canopy_cover", "stress_percentage",
            "yield_estimate", "created_at",
            "processed", "model_version"
        ]

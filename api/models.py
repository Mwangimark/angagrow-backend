from django.db import models


class AnalysisSession(models.Model):
    session_id = models.AutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Aggregated metrics (calculated after all images are processed)
    canopy_cover = models.FloatField(null=True, blank=True)
    stress_percentage = models.FloatField(null=True, blank=True)
    yield_estimate = models.FloatField(null=True, blank=True)

    # NDVI-like indices (averages)
    vari = models.FloatField(null=True, blank=True)
    gli = models.FloatField(null=True, blank=True)
    exg = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Session {self.id} - {self.created_at}"


class DroneImage(models.Model):
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name="images")

    image = models.ImageField(upload_to="drone_images/")
    timestamp = models.DateTimeField(auto_now_add=True)

    # Phase 2 metrics
    vari = models.FloatField(null=True, blank=True)
    gli = models.FloatField(null=True, blank=True)
    exg = models.FloatField(null=True, blank=True)
    canopy_cover = models.FloatField(null=True, blank=True)
    stress_percentage = models.FloatField(null=True, blank=True)

    # Phase 3 metrics
    yield_estimate = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id} in session {self.session.id}"

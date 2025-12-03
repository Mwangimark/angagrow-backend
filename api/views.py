from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from .utils import analyze_drone_image, estimate_yield, generate_recommendations
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .models import AnalysisSession, DroneImage

class CropAnalysisView(APIView):
    def post(self, request):
        images = request.FILES.getlist("images")  # accept multiple images
        if not images:
            return Response({"error": "No images uploaded"}, status=400)

        # Create a new session
        session = AnalysisSession.objects.create()

        # To store aggregated values
        canopy_list, stress_list, yield_list = [], [], []
        vari_list, gli_list, exg_list = [], [], []

        for image_file in images:
            # Save temporarily
            file_path = default_storage.save(f"drone_images/{image_file.name}", image_file)
            full_path = default_storage.path(file_path)

            # Analyze image
            results = analyze_drone_image(full_path)
            if results is None:
                continue  # skip failed images
            
            

            # Estimate yield
            estimate = estimate_yield(results["canopy_pct"], results["stress_pct"])
            yield_estimate = estimate["yield_estimate"]

            # Save individual image analysis
            drone_image = DroneImage.objects.create(
                session=session,
                image=file_path,
                vari=results.get("vari"),
                gli=results.get("gli"),
                exg=results.get("exg"),
                canopy_cover=results.get("canopy_pct"),
                stress_percentage=results.get("stress_pct"),
                yield_estimate=yield_estimate,
            )

            # Add metrics to lists for aggregation
            canopy_list.append(results.get("canopy_pct", 0))
            stress_list.append(results.get("stress_pct", 0))
            yield_list.append(yield_estimate)
            vari_list.append(results.get("vari", 0))
            gli_list.append(results.get("gli", 0))
            exg_list.append(results.get("exg", 0))

        # Aggregate session metrics
        if canopy_list:
            session.avg_canopy_cover = sum(canopy_list) / len(canopy_list)
            session.avg_stress_percentage = sum(stress_list) / len(stress_list)
            session.avg_yield_estimate = sum(yield_list) / len(yield_list)
            session.avg_vari = sum(vari_list) / len(vari_list)
            session.avg_gli = sum(gli_list) / len(gli_list)
            session.avg_exg = sum(exg_list) / len(exg_list)
            session.save()
        
        # Build dictionary for recommendation system
        analysis_summary = {
            "avg_canopy_cover": session.avg_canopy_cover,
            "avg_stress_percentage": session.avg_stress_percentage,
            "avg_yield_estimate": session.avg_yield_estimate,
            "avg_vari": session.avg_vari,
            "avg_gli": session.avg_gli,
            "avg_exg": session.avg_exg,
        }
        # Generate recommendations
        recommendations = generate_recommendations(analysis_summary)
        
        return Response({
            "session_id": session.session_id,
            "num_images_processed": len(canopy_list),
            "avg_canopy_cover": round(session.avg_canopy_cover, 2),
            "avg_stress_percentage": round(session.avg_stress_percentage, 2),
            "avg_yield_estimate": round(session.avg_yield_estimate, 2),
            # NDVI-like indices
            "avg_vari": session.avg_vari,
            "avg_gli": session.avg_gli,
            "avg_exg": session.avg_exg,
            "recommendations": recommendations
        }, status=200)


# chatbot view
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

class ChatbotView(APIView):
    def post(self, request):
        user_message = request.data.get("message")
        if not user_message:
            return Response({"error": "No message provided"}, status=400)

        # Get latest analyzed session
        latest_session = AnalysisSession.objects.order_by("-created_at").first()
        
        if latest_session:
            images_count = DroneImage.objects.filter(session=latest_session).count()

            latest_data = {
                "avg_canopy_cover": round(float(latest_session.avg_canopy_cover), 2),
                "avg_stress_percentage": round(float(latest_session.avg_stress_percentage), 2),
                "avg_yield_estimate": round(float(latest_session.avg_yield_estimate), 2),
                "avg_vari": round(float(latest_session.avg_vari), 3),
                "avg_exg": round(float(latest_session.avg_exg), 3),
                "avg_gli": round(float(latest_session.avg_gli), 3),
                "images_count": images_count
            }
        else:
            latest_data = {}
        
        # Better, smarter prompt
        prompt = f"""
You are a friendly and knowledgeable farming assistant AI.

- Always answer in natural language.
- Do NOT output JSON or raw data.
- When the user asks about yield, stress, canopy, or vegetation, use the values from the latest drone analysis session.
- If the user asks a general question, answer normally but include insights from the latest drone data when helpful.

User message: "{user_message}"

Latest drone analysis data:
{latest_data}

Now provide a clear, helpful, natural-language response.
"""

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs, max_new_tokens=120, temperature=0.7, do_sample=False
        )

        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return Response({"response": bot_response})

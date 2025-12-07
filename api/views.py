from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from .utils import analyze_drone_image, estimate_yield, generate_recommendations
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .models import AnalysisSession, DroneImage
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.core.cache import cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class CropAnalysisView(APIView):
    authentication_classes = [JWTAuthentication]  # Add this line
    permission_classes = [IsAuthenticated]

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
            session.canopy_cover = sum(canopy_list) / len(canopy_list)
            session.stress_percentage = sum(stress_list) / len(stress_list)
            session.yield_estimate = sum(yield_list) / len(yield_list)
            session.vari = sum(vari_list) / len(vari_list)
            session.gli = sum(gli_list) / len(gli_list)
            session.exg = sum(exg_list) / len(exg_list)
            session.save()
        
        # Build dictionary for recommendation system
        analysis_summary = {
            "canopy_cover": session.canopy_cover,
            "stress_percentage": session.stress_percentage,
            "yield_estimate": session.yield_estimate,
            "vari": session.vari,
            "gli": session.gli,
            "exg": session.exg,
        }
        # Generate recommendations
        recommendations = generate_recommendations(analysis_summary)
        
        return Response({
            "session_id": session.session_id,
            "num_images_processed": len(canopy_list),
            "canopy_cover": round(session.canopy_cover, 2),
            "stress_percentage": round(session.stress_percentage, 2),
            "yield_estimate": round(session.yield_estimate, 2),
            # NDVI-like indices
            "vari": session.vari,
            "gli": session.gli,
            "exg": session.exg,
            "recommendations": recommendations
        }, status=200)




# Global model cache
MODEL_CACHE_KEY = "flan_t5_model"
TOKENIZER_CACHE_KEY = "flan_t5_tokenizer"

class ChatbotView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        user_message = request.data.get("message", "").strip()
        if not user_message:
            return Response({"error": "No message provided"}, status=400)

        # Get user context
        user = request.user
        user_role = user.role if hasattr(user, 'role') else 'farmer'
        
        # Get latest analyzed session
        latest_session = AnalysisSession.objects.order_by("-created_at").first()
        
        # DEBUG: Print session details
        print(f"\n{'='*50}")
        print(f"DEBUG - Latest Session:")
        print(f"Session ID: {latest_session.session_id if latest_session else 'None'}")
        if latest_session:
            print(f"Canopy Cover: {latest_session.canopy_cover}")
            print(f"Stress %: {latest_session.stress_percentage}")
            print(f"Yield Estimate: {latest_session.yield_estimate}")
            print(f"VARI: {latest_session.vari}")
            print(f"ExG: {latest_session.exg}")
            print(f"GLI: {latest_session.gli}")
        print(f"{'='*50}\n")
        
        # Prepare context data
        context_data = self.prepare_context(latest_session, user_role)
        
        # DEBUG: Print context data
        print(f"\n{'='*50}")
        print(f"DEBUG - Context Data:")
        for key, value in context_data.items():
            print(f"{key}: {value}")
        print(f"{'='*50}\n")
        
        # Check for quick responses first
        quick_response = self.check_quick_responses(user_message, context_data)
        if quick_response:
            return Response({
                "response": quick_response,
                "context_used": bool(latest_session),
                "user_role": user_role,
                "debug": context_data  # Add debug info to response
            })
        
        # Generate response
        bot_response = self.generate_response_with_timeout(user_message, context_data)
        
        return Response({
            "response": bot_response,
            "context_used": bool(latest_session),
            "user_role": user_role,
            "debug": context_data  # Add debug info to response
        })
    

    def get_model_and_tokenizer(self):
        """Get or load model and tokenizer with caching"""
        model = cache.get(MODEL_CACHE_KEY)
        tokenizer = cache.get(TOKENIZER_CACHE_KEY)
        
        if not model or not tokenizer:
            print("Loading model and tokenizer...")
            # Use a smaller model for faster responses
            model_name = "google/flan-t5-small"  # Smaller and faster than base
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Cache for future requests
            cache.set(MODEL_CACHE_KEY, model, timeout=3600)  # 1 hour cache
            cache.set(TOKENIZER_CACHE_KEY, tokenizer, timeout=3600)
        
        return model, tokenizer

    def check_quick_responses(self, user_message, context):
        """Handle common questions without model inference for speed"""
        user_message_lower = user_message.lower()
        
        quick_responses = {
            "hello": "Hello! I'm your AngaGrow AI assistant. How can I help with your farm analysis today?",
            "hi": "Hi there! I'm ready to help with your crop analysis and farming questions.",
            "thanks": "You're welcome! Let me know if you need any more assistance.",
            "thank you": "You're welcome! Happy to help with your farming needs.",
            "what is your name": "I'm AngaGrow AI, your intelligent farming assistant!",
            "who are you": "I'm AngaGrow AI, your expert farming assistant powered by drone analysis and AI.",
        }
        
        # Check for exact matches
        if user_message_lower in quick_responses:
            return quick_responses[user_message_lower]
        
        # Check for patterns
        if "canopy" in user_message_lower or "cover" in user_message_lower:
            return self.generate_canopy_response(context)
        elif "stress" in user_message_lower:
            return self.generate_stress_response(context)
        elif "yield" in user_message_lower or "harvest" in user_message_lower:
            return self.generate_yield_response(context)
        elif "fertilizer" in user_message_lower or "fertilize" in user_message_lower:
            return self.generate_fertilizer_response(context)
        
        return None

    def generate_canopy_response(self, context):
        """Generate canopy-specific response"""
        if context["has_data"]:
            canopy = context["canopy_cover"]
            if canopy > 80:
                status = "excellent"
                advice = "Your crops are dense and healthy. Maintain current practices."
            elif canopy > 60:
                status = "good"
                advice = "Your canopy is developing well. Consider light fertilization if needed."
            else:
                status = "needs attention"
                advice = "Your canopy could be denser. Check soil nutrients and irrigation."
            
            return f"Your canopy cover is {canopy}%, which is {status}. Canopy cover measures how much ground is covered by crop leaves. {advice} For optimal growth, aim for 70-90% canopy cover."
        else:
            return "Canopy cover measures the percentage of ground covered by crop leaves. Ideal canopy cover is 70-90%. Without your drone data, I can't give specific advice, but generally ensure proper spacing, adequate nutrients, and regular irrigation."

    def generate_stress_response(self, context):
        """Generate stress-specific response"""
        if context["has_data"]:
            stress = context["stress_level"]
            if stress < 15:
                status = "low"
                advice = "Your crops are healthy. Continue current management."
            elif stress < 30:
                status = "moderate"
                advice = "Monitor closely. Check water levels and look for pests."
            else:
                status = "high"
                advice = "Take immediate action. Review irrigation, fertilization, and pest control."
            
            return f"Your crop stress level is {stress}%, which is {status}. Stress can come from water, nutrients, or pests. {advice} Ideal stress levels should be below 20%."
        else:
            return "Crop stress can result from water deficiency, nutrient imbalance, pests, or disease. Common signs include wilting, yellowing, or stunted growth. To reduce stress, ensure consistent irrigation, balanced fertilization, and regular pest monitoring."

    def generate_yield_response(self, context):
        """Generate yield-specific response"""
        if context["has_data"]:
            yield_est = context["yield_estimate"]
            if yield_est > 5:
                rating = "excellent"
                advice = "You're on track for a great harvest!"
            elif yield_est > 3:
                rating = "good"
                advice = "Your yield is promising. Continue good practices."
            else:
                rating = "needs improvement"
                advice = "Consider reviewing your fertilization and irrigation strategies."
            
            return f"Your estimated yield is {yield_est} tons per hectare, which is {rating}. {advice} For most crops, optimal yield ranges from 3-6 tons/ha depending on the crop type and conditions."
        else:
            return "Yield estimates predict how much crop you'll harvest. Without your specific drone data, I recommend: 1) Ensure proper spacing, 2) Maintain soil fertility, 3) Control pests and diseases, 4) Time irrigation correctly. Harvest when crops reach maturity and weather conditions are dry."

    def generate_fertilizer_response(self, context):
        """Generate fertilizer-specific response"""
        if context["has_data"]:
            vari = context["vari_index"]
            if vari > 0.6:
                recommendation = "Use a balanced NPK fertilizer (10-10-10) at maintenance levels."
            elif vari > 0.4:
                recommendation = "Apply nitrogen-rich fertilizer to boost growth."
            else:
                recommendation = "Use a complete fertilizer mix and consider soil testing."
            
            return f"Based on your VARI index of {vari}, I recommend: {recommendation} Apply fertilizer in the morning and water thoroughly afterward. Avoid over-fertilization as it can harm crops."
        else:
            return "For fertilizer recommendations: 1) Test your soil first, 2) Use balanced NPK fertilizer, 3) Apply during growth stages, 4) Avoid excessive nitrogen. Different crops need different nutrients at various growth stages."

    def generate_response_with_timeout(self, user_message, context, timeout_seconds=3):
        """Generate response with timeout to prevent hanging"""
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def generate_thread():
            try:
                response = self.generate_response(user_message, context)
                result_queue.put(response)
            except Exception as e:
                result_queue.put(f"I encountered an issue. {self.get_fallback_response(user_message, context)}")
        
        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            # Thread timed out, return quick response
            return f"I'm processing your question about '{user_message[:50]}...'. While I analyze, here's quick advice: {self.get_quick_advice(user_message)}"
        
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return self.get_fallback_response(user_message, context)

    def generate_response(self, user_message, context):
        """Generate intelligent response using FLAN-T5 with optimization"""
        
        # Build prompt
        prompt = self.build_prompt(user_message, context)
        
        try:
            # Get model and tokenizer
            model, tokenizer = self.get_model_and_tokenizer()
            
            # Tokenize with optimized settings
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=256,  # Reduced from 512
                truncation=True
            )
            
            # Generate with optimized parameters
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,  # Reduced from 200
                temperature=0.7,
                do_sample=False,  # Changed to False for speed
                top_p=0.85,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                length_penalty=0.8,
                early_stopping=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            response = self.clean_response(response, user_message)
            
            # Ensure minimum length
            if len(response.split()) < 10:
                response = f"{response} {self.get_quick_advice(user_message)}"
            
            return response
            
        except Exception as e:
            print(f"Model error: {e}")
            return self.get_fallback_response(user_message, context)

    def build_prompt(self, user_message, context):
        """Build optimized prompt"""
        if context["has_data"]:
            context_summary = f"""
Data: Canopy {context['canopy_cover']}%, Stress {context['stress_level']}%, Yield {context['yield_estimate']}t/ha.
User role: {context['user_role']}. Date: {context['analysis_date']}.
"""
        else:
            context_summary = "No specific farm data available. User role: {context['user_role']}."
        
        prompt = f"""Answer as AngaGrow AI farming assistant. Be concise.

Context: {context_summary}

Question: {user_message}

Answer briefly and helpfully:"""
        
        return prompt

    def get_quick_advice(self, user_message):
        """Get quick advice for timeout situations"""
        advice_map = {
            "canopy": "Aim for 70-90% canopy cover through proper spacing and nutrition.",
            "stress": "Reduce stress with consistent irrigation and pest control.",
            "yield": "Optimize yield with timely planting and balanced fertilization.",
            "fertilizer": "Use soil testing to determine exact fertilizer needs.",
        }
        
        user_lower = user_message.lower()
        for key, advice in advice_map.items():
            if key in user_lower:
                return advice
        
        return "Review your farming practices and consider drone analysis for precise recommendations."

    def clean_response(self, response, user_message):
        """Clean response quickly"""
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        # Simple cleanup
        response = response.strip()
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response[:500]  # Limit response length

    def get_fallback_response(self, user_message, context):
        """Quick fallback responses"""
        fallbacks = [
            f"As your {context.get('user_role', 'farming')} assistant, I suggest checking your latest drone analysis for precise advice on '{user_message}'.",
            f"For '{user_message}', I recommend consulting your farm data or uploading new drone images for analysis.",
            f"I can help with '{user_message}'. Please ensure your drone data is uploaded for personalized advice.",
        ]
        
        import random
        return random.choice(fallbacks)

    def prepare_context(self, session, user_role):
        """Prepare context data (unchanged)"""
        if not session:
            return {
                "has_data": False,
                "user_role": user_role,
                "message": "No drone analysis data available yet."
            }
        
        images_count = DroneImage.objects.filter(session=session).count()
        
        return {
            "has_data": True,
            "user_role": user_role,
            "images_analyzed": images_count,
            "canopy_cover": round(float(session.canopy_cover), 2) if session.canopy_cover else 0,
            "stress_level": round(float(session.stress_percentage), 2) if session.stress_percentage else 0,
            "yield_estimate": round(float(session.yield_estimate), 2) if session.yield_estimate else 0,
            "vari_index": round(float(session.vari), 3) if session.vari else 0,
            "exg_index": round(float(session.exg), 3) if session.exg else 0,
            "gli_index": round(float(session.gli), 3) if session.gli else 0,
            "analysis_date": session.created_at.strftime("%Y-%m-%d") if session.created_at else "Unknown"
        }
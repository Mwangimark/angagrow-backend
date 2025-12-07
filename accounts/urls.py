from django.urls import path
from .views import RegisterView
from rest_framework_simplejwt.views import TokenRefreshView as SimpleJWTTokenRefreshView
from .views import LoginView, LogoutView, UserProfileView, TokenRefreshView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    
    # Authentication
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    
    # Profile
    path('profile/', UserProfileView.as_view(), name='profile'),
    
    # Token refresh (you can use either)
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh_custom'),

]

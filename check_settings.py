
from config.settings import get_settings
import os

settings = get_settings()
print(f"NVIDIA_NIM_API_KEY: '{settings.nvidia_nim_api_key}'")
print(f"OPENROUTER_API_KEY: '{settings.open_router_api_key}'")
print(f"DEEPSEEK_API_KEY: '{settings.deepseek_api_key}'")
print(f"MODEL: '{settings.model}'")
print(f"ENV NVIDIA_NIM_API_KEY: '{os.environ.get('NVIDIA_NIM_API_KEY')}'")

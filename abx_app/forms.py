from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm


class SignupForm(UserCreationForm):
    class Meta:
        model = get_user_model()
        fields = ["username", "password1", "password2"]


class LoginForm(AuthenticationForm):
    pass


class MonitorSizeForm(forms.Form):
    monitor_size = forms.FloatField(label="Enter your monitor size (inch)")

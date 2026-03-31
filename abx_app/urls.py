from django.urls import path

from . import views

urlpatterns = [
    path("", views.login_view, name="login"),
    path("home/", views.home_view, name="home"),
    path("signup/", views.signup_view, name="signup_view"),
    path("monitor/", views.monitor_view, name="monitor"),
    path("introduction/", views.introduction_view, name="introduction"),
    path("introduction_end/", views.introduction_end_view, name="introduction_end"),
    path("exp_material_view/", views.exp_material_view, name="exp_material_view"),
    path(
        "exp_material_end_view/",
        views.exp_material_end_view,
        name="exp_material_end_view",
    ),
    path("logout/", views.logout_view, name="logout"),
    path("end/", views.end_view, name="end"),
]

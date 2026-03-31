from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
from import_export import resources
from import_export.admin import ImportExportModelAdmin

from .models import ResultsABX, User

# Register your models here.


class UserResource(resources.ModelResource):

    # Hash the password during import
    def before_import_row(self, row, row_number=None, **kwargs):
        value = row["password"]
        row["password"] = make_password(value)

    class Meta:
        model = get_user_model()


User = get_user_model()


@admin.register(User)
class UserAdmin(ImportExportModelAdmin):
    ordering = ["id"]
    list_display = ("id", "username", "last_login", "task_count")

    resource_class = UserResource


@admin.register(ResultsABX)
class ResultsABXAdmin(admin.ModelAdmin):
    ordering = ["id"]
    list_display = (
        "id",
        "start_datetime",
        "user",
        "end_datetime",
        "average_hit",
        "answer_hit",
        "answer_target",
        "trial_response",
        "trial_rt",
        "trial_count",
        "indices_target",
        "first_stim",
    )

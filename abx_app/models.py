from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractUser
# Create your models here.
from django.db import models


class User(AbstractUser):
    # experiment config
    config_exp = models.JSONField(null=True, blank=True)
    list_conditions = models.JSONField(default=list)
    list_conditions_name = models.JSONField(default=list)
    _list_conditions_name_copy = models.JSONField(default=list) # used for next shuffle
    list_convergence = models.JSONField(default=list)

    length_conditions = models.IntegerField(default=0)
    count_cond = models.IntegerField(default=0)

    # repeat param
    task_count = models.IntegerField(default=0)

    # image path for the current trial (now) and the next trial (next)
    path_now = models.TextField(blank=True)
    path_next = models.TextField(blank=True)

    # image pooled list
    list_imgs = models.JSONField(default=list)

    # flag for eye fixation
    is_fixate = models.BooleanField(default=True)

    # count the number of eye missing
    miss_eye_position = models.IntegerField(default=0)


    # Dictionary for storing results from one experimental block, reinitialized to match the number of conditions in config_abx.yaml
    dict_res_threshold = models.JSONField(default=dict)

    # Dictionary for storing all results of the participant; used to check for unfinished keys at experiment start
    dict_all_res_threshold = models.JSONField(default=dict) # target threshold
    dict_all_res_actual_value = models.JSONField(default=dict) # actual value

    # whether each condition was converged
    dict_all_res_convergence = models.JSONField(default=dict)
    dict_all_res_flip_count = models.JSONField(default=dict)
    dict_all_res_current_stair = models.JSONField(default=dict)

    # Staircase sequence of each condition. ideal threshold, staircase flip, actual value
    dict_all_sequence_threshold = models.JSONField(default=dict)
    dict_all_sequence_hit = models.JSONField(default=dict)
    dict_all_sequence_actual_value = models.JSONField(default=dict)
    dict_all_sequence_flip = models.JSONField(default=dict)


    # Temporary buffer used for staircase updates (2up/1down)
    dict_all_responses_buffer = models.JSONField(default=dict)


class MonitorSize(models.Model):
    size = models.FloatField()  # Field for storing the monitor size


class ResultsABX(models.Model):
    # user information
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)

    # exp time information
    start_datetime = models.TextField(blank=True)
    end_datetime = models.TextField(blank=True)

    # experiment results
    answer_hit = models.TextField(blank=True)
    answer_target = models.TextField(blank=True)
    trial_response = models.TextField(blank=True)
    trial_rt = models.TextField(blank=True)
    trial_count = models.TextField(blank=True)
    first_stim = models.TextField(blank=True)
    indices_target = models.TextField(blank=True)
    average_hit = models.FloatField(blank=True, default=0)

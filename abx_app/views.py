import copy
import os
from pathlib import Path
import numpy as np
import time
import random
from datetime import datetime
from multiprocessing import Process, set_start_method
import yaml
from django.contrib.auth import get_user_model, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .forms import LoginForm, MonitorSizeForm, SignupForm
from .models import MonitorSize, ResultsABX
from .tasks import load_actual_value, worker_function
from .utils import init_conditions, update_staircase
from .AttackCNN.utils.condition import Condition

# config import
with open("abx_app/config_abx.yaml", "r") as f:
    config_abx = yaml.safe_load(f)
if config_abx["STIM"]["IS_TOBII"]:
    from .eye_tracks import start_eyetrack, stop_eyetrack, check_gaze_within_central_region
# parallel process of cuda
set_start_method("spawn")


# User registration form
def signup_view(request):
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(to="login")

    else:
        form = SignupForm()

    param = {"form": form}
    return render(request, "abx_app/signup.html", param)


# Login form
def login_view(request):
    if request.method == "POST":
        form = LoginForm(request, data=request.POST)

        if form.is_valid():
            user = form.get_user()

            if user:
                login(request, user)

                # register config to database
                user.config_exp = config_abx

                # initialize the experimental condition
                user = init_conditions(user, user.config_exp["EXP_CONDS"])


                # make path_now
                # make int to avoid cash problem
                now = datetime.now()
                unique_int = int(now.strftime("%Y%m%d%H%M%S"))
                user.path_now = f"{user.username}/trial_{unique_int}_{user.task_count}"
                os.makedirs(user.config_exp["PATH"]["ROOT_IMG"] + user.username, exist_ok=True)
                user.save()

                name_current_cond = user.list_conditions_name[user.count_cond]
                cond = Condition.from_string(name_current_cond)
                threshold = user.dict_res_threshold[name_current_cond]
                print(name_current_cond)

                process = Process(
                    target=worker_function,
                    kwargs={
                        "cond": cond,
                        "threshold": threshold,
                        "path_save": user.config_exp["PATH"]["ROOT_IMG"]
                        + user.path_now,
                    },
                )
                process.start()

                if user.task_count == 0:
                    return redirect(to="introduction")
                else:
                    return redirect(to="exp_material_view")

    else:
        form = LoginForm()

    param = {"form": form}
    return render(request, "abx_app/login.html", param)


# Logout page
@login_required
def logout_view(request):
    logout(request)
    return render(request, "abx_app/logout.html")


# Home page
@login_required
def home_view(request):
    user = request.user
    user.task_count = 0
    user.save()
    param = {"user": user}
    return render(request, "abx_app/home.html", param)


# About page
@login_required
def aboutapp_view(request):
    return render(request, "abx_app/index_aboutapp.html")


@login_required
def monitor_view(request):
    if request.method == "POST":
        form = MonitorSizeForm(request.POST)
        if form.is_valid():
            monitor_size = form.cleaned_data["monitor_size"]
            MonitorSize.objects.create(size=monitor_size)  # Save to the model
            return redirect("introduction")  # Set the redirect destination
    else:
        form = MonitorSizeForm()

    return render(request, "abx_app/index_monitor.html", {"form": form})


@login_required
def introduction_view(request):
    user = request.user

    # check if the initial image is already prepared
    while not os.path.exists(user.config_exp["PATH"]["ROOT_IMG"] + user.path_now + "_fake.jpg"):
        print("Wait for image generation")
        time.sleep(0.2)

    # Move to the preliminary instruction page
    param = {
        "user": user,
    }

    return render(request, "abx_app/index_introduction.html", param)


# End of the first instruction page
@login_required
def introduction_end_view(request):
    return redirect(to="exp_material_view")


@login_required
def exp_material_view(request):
    # make int to avoid cash problem
    now = datetime.now()
    unique_int = int(now.strftime("%Y%m%d%H%M%S"))

    # load current user info
    user = request.user
    # make path_next
    user.path_next = f"{user.username}/trial_{unique_int}_{user.task_count + 1}"

    # make images and save them for the next trial as async process
    if user.count_cond == user.length_conditions-1:
        ind_cond = 0
        user._list_conditions_name_copy = copy.deepcopy(user.list_conditions_name)
        random.shuffle(user._list_conditions_name_copy)
        name_next_cond = user._list_conditions_name_copy[ind_cond]
        # Update to the shuffled condition-name list after the trial
    else:
        ind_cond = user.count_cond + 1
        name_next_cond = user.list_conditions_name[ind_cond]

    print(user.path_next)
    user.save()

    cond = Condition.from_string(name_next_cond)
    threshold = user.dict_res_threshold[name_next_cond]
    print(f'next condition is {name_next_cond}')

    process = Process(
        target=worker_function,
        kwargs={
            "cond": cond,
            "threshold": threshold,
            "path_save": user.config_exp["PATH"]["ROOT_IMG"] + user.path_next,
        },
    )
    process.start()

    if user.is_fixate:
        if (user.task_count) % (user.config_exp["REP"]["NUM_PER_READY"]) == 0:
            flag_ready = "ready"
        else:
            flag_ready = "false"
    else:
        flag_ready = "eye"


    # Here, the image presentation is based on path_now
    param = {
        "user": user.username,
        "path_img_pre": user.path_now,
        "APP_VER": unique_int,
        "flag_ready": flag_ready,
        "size_img": user.config_exp["STIM"]["SIZE_IMG"],
        "shift_img": user.config_exp["EXP_CONDS"]["ECC_COND"][0],
    }

    if user.config_exp["STIM"]["IS_TOBII"]:
        #eye tracking start
        start_eyetrack()

    return render(request, "abx_app/index_exp.html", param)


@login_required
def exp_material_end_view(request):
    if request.method == "POST":
        results = ResultsABX()
        # Save the user
        username = request.POST.get("userName")
        user = get_user_model().objects.get(username=username)
        results.user = user

        # Save the gaze start time
        results.start_datetime = request.POST.get("gazeStartTime")

        # Save the gaze end time
        results.end_datetime = request.POST.get("gazeEndTime")

        # Save the results
        results.answer_hit = request.POST.get("answer_hit")
        results.answer_target = request.POST.get("answer_target")
        results.trial_response = request.POST.get("trial_response")
        results.trial_rt = request.POST.get("trial_rt")
        results.trial_count = request.POST.get("trial_count")
        results.first_stim = request.POST.get("first_stim")
        results.indices_target = request.POST.get("indices_target")
        results.average_hit = float(request.POST.get("average_hit"))

        results.save()
        print(f"accuarcy is: {results.average_hit}")

    user = request.user
    print(f'eye start time {results.start_datetime} end time {results.end_datetime}')

    # get actual value
    actual_value = load_actual_value(user.list_conditions_name[user.count_cond],
                                     user.config_exp["PATH"]["ROOT_IMG"] + user.path_now
                                     )

    if user.config_exp["STIM"]["IS_TOBII"]:
        # eye tracking record stop
        data_eye, time_eye = stop_eyetrack()

        # parameters are updated when eye position was fixated at the center
        if check_gaze_within_central_region(data_eye, time_eye,
                                            np.array(results.start_datetime).astype('float')/1000,
                                            np.array(results.end_datetime).astype('float')/1000,
                                            user.config_exp["STIM"]["RADIUS_THRESHOLD"],
                                            user.config_exp["STIM"]["MONITOR_RES_W"],
                                            user.config_exp["STIM"]["MONITOR_RES_H"],
                                            ):
            user = update_staircase(user, results.average_hit, float(actual_value))
            print('eye position was okay')
            user.is_fixate = True
        else:
            print('eye position was not in the center')
            user.is_fixate = False
            user.miss_eye_position += 1
            print(f'the number of missing eye position is {user.miss_eye_position}')
    else:
        user = update_staircase(user, results.average_hit, float(actual_value))

    user.task_count += 1

    print(user.list_conditions_name)
    if user.count_cond == user.length_conditions-1:
        user.count_cond = 0
        user.list_conditions_name = copy.deepcopy(user._list_conditions_name_copy)
        print(f'the next block order is {user.list_conditions_name}')
    else:
        user.count_cond += 1

    # update path_now for the next trial
    # loop until the image generation is finished
    while not os.path.exists(user.config_exp["PATH"]["ROOT_IMG"] + user.path_next + "_fake.jpg"):
        print("Wait for image generation")
        time.sleep(0.2)
    user.path_now = copy.deepcopy(user.path_next)
    user.save()

    if user.task_count == user.config_exp["REP"]["REP_MAX"]:
        return redirect(to="end")
    else:
        return redirect(to="exp_material_view")


@login_required
def end_view(request):
    logout(request)
    return render(request, "abx_app/index_end.html")

import os
import glob
import random
from itertools import product

from .AttackCNN.utils.condition import Condition



def init_conditions(user, config_cond):
    # make conditions from yaml config
    user.list_conditions_name, user.list_convergence = make_conditions_from_lists(config_cond)
    random.shuffle(user.list_conditions_name)
    print(f'the next block order is {user.list_conditions_name}')
    #register the experimental condition
    user.length_conditions = len(user.list_conditions_name)
    user.count_cond = 0



    # init target threshold
    user.dict_res_threshold = make_results_dict(user.list_conditions_name, config_cond['INIT_VAL'], all_dict=user.dict_all_res_threshold,
                                                is_reset=config_cond["IS_RESET"])



    user.dict_all_sequence_threshold = make_sequence_dict(user.list_conditions_name, [], all_dict=user.dict_all_sequence_threshold,
                                                is_reset=config_cond["IS_RESET"])
    user.dict_all_sequence_hit = make_sequence_dict(user.list_conditions_name, [], all_dict=user.dict_all_sequence_hit,
                                                is_reset=config_cond["IS_RESET"])
    user.dict_all_sequence_flip = make_sequence_dict(user.list_conditions_name, [], all_dict=user.dict_all_sequence_flip,
                                                is_reset=config_cond["IS_RESET"])
    user.dict_all_sequence_actual_value = make_sequence_dict(user.list_conditions_name, [], all_dict=user.dict_all_sequence_actual_value,
                                                is_reset=config_cond["IS_RESET"])

    user.dict_all_responses_buffer = make_sequence_dict(user.list_conditions_name, [], all_dict=user.dict_all_responses_buffer,
                                                is_reset=config_cond["IS_RESET"])

    user.task_count = 0

    return user

def make_conditions_from_lists(config_cond):
    # assume there are LAYER_COND, COMP_COND, SIGN_COND
    # make combinations of them
    list_conditions = list(product(config_cond['ECC_COND'], config_cond['MODE'], config_cond['LAYER_COND'], config_cond['COMP_COND'], config_cond['SIGN_COND']))
    # get the condition names in str
    list_conditions_name = [Condition(ecc=ecc, mode=mode, layer=layer, component=component, direction=direction).to_string() for (ecc, mode, layer, component, direction) in list_conditions]
    # make a boolean list for checking the convergence of up down method
    list_convergence = [False for _ in range(len(list_conditions_name))]

    return list_conditions_name, list_convergence

def make_results_dict(list_keys, dict_init, all_dict={}, is_reset=False):
    init_dict = {}
    for key in list_keys:
        print(key)
        cond = Condition.from_string(key)
        if is_reset:
            print(dict_init)
            if cond.layer in dict_init:
                init_dict[key] = dict_init[cond.layer]
            else:
                raise KeyError(f"Key '{cond.layer}' not found in dict_init of config_abx.yaml.")
        else:
            if key not in all_dict:
                if cond.layer in dict_init:
                    init_dict[key] = dict_init[cond.layer]
                else:
                    raise KeyError(f"Key '{cond.layer}' not found in dict_init of config_abx.yaml.")
            else:
                init_dict[key] = all_dict[key]

    return init_dict

def make_sequence_dict(list_keys, val_init, all_dict={}, is_reset=False):
    for key in list_keys:
        if is_reset:
            all_dict[key] = val_init
        else:
            if not key in all_dict:
                all_dict[key] = val_init
    return all_dict


def update_staircase(user, average_hit, actual_value):
    name_cond = user.list_conditions_name[user.count_cond]
    cond = Condition.from_string(name_cond)
    step_val = user.config_exp["EXP_CONDS"]["STEP_VAL"][cond.layer]
    min_val = user.config_exp["EXP_CONDS"]["MIN_VAL"][cond.layer]

    # save results to all lists
    ## target threhold
    user.dict_all_sequence_threshold[name_cond].append(user.dict_res_threshold[name_cond])
    ## actual value
    user.dict_all_res_actual_value[name_cond] = actual_value
    user.dict_all_sequence_actual_value[name_cond].append(actual_value)
    ## hit
    user.dict_all_sequence_hit[name_cond].append(average_hit)

    tmp_list = user.dict_all_responses_buffer[name_cond]
    tmp_list.append(average_hit)
    if not name_cond in user.dict_all_res_flip_count:
        user.dict_all_res_flip_count[name_cond] = 0
    if not name_cond in user.dict_all_res_current_stair:
        user.dict_all_res_current_stair[name_cond] = 'keep'
    # assume 2up 1down
    if average_hit == 1:
        if len(tmp_list) > 1:
            if tmp_list[-1] + tmp_list[-2] == 2:
                print("down")
                user.dict_res_threshold[name_cond] -= step_val
                # check if threshold is minimum
                if user.dict_res_threshold[name_cond] < min_val:
                    user.dict_res_threshold[name_cond] = min_val

                user.dict_all_responses_buffer[name_cond] = []
                # check and count the number of staircase flip
                if user.dict_all_res_current_stair[name_cond] == 'down':
                    user.dict_all_sequence_flip[name_cond].append(False)
                else:
                    user.dict_all_sequence_flip[name_cond].append(True)
                    user.dict_all_res_flip_count[name_cond] += 1
                    user.dict_all_res_current_stair[name_cond] = 'down'

            else:
                print("keep")
                user.dict_all_responses_buffer[name_cond] = tmp_list
                user.dict_all_sequence_flip[name_cond].append(False)
        else:
            print("keep")
            user.dict_all_responses_buffer[name_cond] = tmp_list
            user.dict_all_sequence_flip[name_cond].append(False)
    else:
        print("up")
        user.dict_res_threshold[name_cond] += step_val
        user.dict_all_responses_buffer[name_cond] = tmp_list
        # check and count the number of staircase flip
        if user.dict_all_res_current_stair[name_cond] == 'up':
            user.dict_all_sequence_flip[name_cond].append(False)
        else:
            user.dict_all_sequence_flip[name_cond].append(True)
            user.dict_all_res_flip_count[name_cond] += 1
            user.dict_all_res_current_stair[name_cond] = 'up'

    if user.dict_all_res_flip_count[name_cond]>=user.config_exp['EXP_CONDS']['NUM_CONVERGE']:
        user.dict_all_res_convergence[name_cond] = True
    else:
        user.dict_all_res_convergence[name_cond] = False

    #update global target threshold
    user.dict_all_res_threshold[name_cond] = user.dict_res_threshold[name_cond]

    return user


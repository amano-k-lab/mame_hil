import tobii_research as tr
import time
import numpy as np

found_eyetrackers  = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]
print("Address: " + my_eyetracker.address)
print("Model: " + my_eyetracker.model)
print("Name (It's OK if this is empty): " + my_eyetracker.device_name)
print("Serial number: " + my_eyetracker.serial_number)

global_gaze_data = []
time_stamp = []

def gaze_data_callback(gaze_data):
   global global_gaze_data
   global time_stamp
   time_stamp.append(time.time())

   lx = gaze_data.left_eye.gaze_point.position_on_display_area[0]
   ly = gaze_data.left_eye.gaze_point.position_on_display_area[1]
   rx = gaze_data.right_eye.gaze_point.position_on_display_area[0]
   ry = gaze_data.right_eye.gaze_point.position_on_display_area[1]
   global_gaze_data.append(((lx+rx)/2,(ly+ry)/2))

def gaze_data(eyetracker):
   #global global_gaze_data
   # Print gaze points of left and right eye
   my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=False)

def start_eyetrack():
    global my_eyetracker
    gaze_data(my_eyetracker)

def stop_eyetrack():
    global my_eyetracker
    global global_gaze_data
    global time_stamp
    # Stop the thread here
    print('finishing...')

    my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    tmp_output = global_gaze_data
    tmp_time = time_stamp
    global_gaze_data = []
    time_stamp =[]
    return tmp_output, tmp_time

def is_within_central_region(gaze_point, RADIUS_THRESHOLD, CENTRAL_POINT_X=0.5, CENTRAL_POINT_Y=0.5, MON_W=1920, MON_H=1080):
    x, y = gaze_point
    distance = np.sqrt((x - CENTRAL_POINT_X*MON_W) ** 2 + (y - CENTRAL_POINT_Y*MON_H) ** 2)
    print(f'distance is {distance}')
    return distance <= RADIUS_THRESHOLD

def check_gaze_within_central_region(data_gaze_output, data_time, lower_bound, upper_bound, RADIUS_THRESHOLD, MON_W, MON_H):
    data_time = np.array(data_time)
    index_time = np.where((data_time >= lower_bound) & (data_time <= upper_bound))[0].tolist()
    data_gaze_output_extracted = [np.array(data_gaze_output[i]) for i in index_time]
    mean_xy = np.nanmean(np.array(data_gaze_output_extracted), axis=0) # ignore eye blink
    # mean_xy = np.mean(np.array(data_gaze_output_extracted), axis=0)
    return is_within_central_region((mean_xy[0]*MON_W,mean_xy[1]*MON_H), RADIUS_THRESHOLD, CENTRAL_POINT_X=0.5, MON_W=MON_W, MON_H=MON_H)


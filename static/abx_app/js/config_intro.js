
// Query string used for cache busting
// See the following site for details
// https://netamame.com/cache-busting/
const APP_VER = '?23v2';

// Project root
const APP_ROOT = '/abx_app/';
// const APP_ROOT = '../'; // for CGI

// Static root
const STATIC_ROOT = '/static/';
// const STATIC_ROOT = '../../static/'; // for CGI

// Image root
const IMAGE_ROOT = 'abx_app/img/';

// Condition image root
const CONDITION_ROOT = 'orig/';
//const CONDITION_ROOT = 'gray/';
//const CONDITION_ROOT = 'reduce_2/';
//const CONDITION_ROOT = 'reduce_4/';

const NUM_REP = 20;

// Target frame rate
// Using the least common multiple worked well for 60Hz, 90Hz, and 120Hz devices
// In one case, setting the target frame rate to 120 caused 45 fps on a 90Hz device
targetFrameRate = 360;

// Canvas size information
const canvas = new CanvasInfo(Math.max(window.innerWidth, window.innerHeight),Math.min(window.innerWidth, window.innerHeight));

const Config = {};
// App background color
Config.BackGround = {};
Config.BackGround.main_color = "#000";
Config.BackGround.task_color = "#000";

// Fixation
Config.Fixation = {};
Config.Fixation.d1 = Math.round(canvas.height * 0.045);
Config.Fixation.d2 = Math.round(canvas.height * 0.015);
Config.Fixation.colorOval = 255;
Config.Fixation.colorCross = 0;

// Monitor
// "monitor_size" is imported from user (in inch)
let viewer_dist = 50;
function get_ppd(viewer_dist, screen_params){
    return (viewer_dist*Math.tan(Math.PI/180)) * screen_params;
}
let window_availw = window.screen.availWidth;
let window_ratio = window.screen.availHeight/window_availw;
let width_screen_cm = Math.sqrt((monitor_size*2.54)**2/(1+window_ratio**2));
let screen_params = window_availw/width_screen_cm;
let ppd = get_ppd(viewer_dist, screen_params);

// Image
Config.Image = {};
Config.Image.arrays_target = [];
Config.Image.arrays_miss = [];
Config.Image.size_rescale = Math.round(5*ppd); //in pixel
Config.Image.shift_left_large = Math.round(11*ppd);  //in pixel
Config.Image.shift_left_small = Math.round(5*ppd);  //in pixel
Config.Image.shift_right_large = Math.round(11*ppd);  //in pixel
Config.Image.shift_right_small = Math.round(5*ppd);  //in pixel
Config.Image.shift_up = Math.round(3*ppd);  //in pixel
Config.Image.shift_down = Math.round(3*ppd);  //in pixel


// Button
Config.Button = {};
Config.Button.width = Math.round(canvas.height * 0.18);
Config.Button.height = Math.round(canvas.height * 0.1);
Config.Button.position = [canvas.center_x - Config.Button.width / 2, canvas.bottom - (canvas.height * 0.1) - (Config.Button.height / 2)];
Config.Button.text_size = Math.round(canvas.height * 0.03);
Config.Button.space_offset = Math.round(canvas.height * 0.25);

// Text
Config.Text = {};
Config.Text.color = 255;
Config.Text.fontSize = canvas.height * 0.05;


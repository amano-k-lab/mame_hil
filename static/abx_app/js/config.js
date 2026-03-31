
// Query string used for cache busting
// See the following site for details
// https://netamame.com/cache-busting/
// const APP_VER = '?23v2';

// Project root
const APP_ROOT = '/abx_app/';
// const APP_ROOT = '../'; // for CGI

// Static root
const STATIC_ROOT = '/static/';
// const STATIC_ROOT = '../../static/'; // for CGI

// Image root
const IMAGE_ROOT = 'abx_app/img/';



// Target frame rate
// Using the least common multiple worked well for 60Hz, 90Hz, and 120Hz devices
// In one case, setting the target frame rate to 120 caused 45 fps on a 90Hz device
targetFrameRate = 360;

// Canvas size information
const canvas = new CanvasInfo(Math.max(window.screen.availWidth, window.screen.availHeight),Math.min(window.screen.availWidth, window.screen.availHeight));

const Config = {};
// App background color
Config.BackGround = {};
Config.BackGround.main_color = "#000";
Config.BackGround.task_color = "#000";



// Monitor
let viewer_dist = 70;
// fixed moniter size (in inch)
let monitor_size = 24;

function get_ppd(viewer_dist, screen_params){
    return (viewer_dist*Math.tan(Math.PI/180)) * screen_params;
}
let window_availw = window.screen.availWidth;
let window_ratio = window.screen.availHeight/window_availw;
//let width_screen_cm = Math.sqrt((monitor_size*2.54)**2/(1+window_ratio**2));
let width_screen_cm = 54;
let screen_params = window_availw/width_screen_cm;
let ppd = get_ppd(viewer_dist, screen_params);
console.log(ppd)

// Fixation
Config.Fixation = {};
Config.Fixation.d1 = Math.round(0.45*ppd);
Config.Fixation.d2 = Math.round(0.15*ppd);
Config.Fixation.colorOval = 255;
Config.Fixation.colorCross = 0;

// Image
Config.Image = {};
Config.Image.arrays_orig = [];
Config.Image.arrays_fake = [];
Config.Image.size_rescale = Math.round(SIZE_IMG*ppd); //in pixel

// shift parameter comes from django
Config.Image.shift_right = Math.round(SHIFT_IMG*ppd);  //in pixel


// Button
Config.Button = {};
Config.Button.width = Math.round(canvas.height * 0.12);
Config.Button.height = Math.round(canvas.height * 0.04);
Config.Button.position = [canvas.center_x - Config.Button.width / 2, canvas.bottom - (canvas.height * 0.1) - (Config.Button.height / 2)];
Config.Button.text_size = Math.round(canvas.height * 0.03);
Config.Button.space_offset = Math.round(canvas.height * 0.25);

// Text
Config.Text = {};
Config.Text.color = 255;
Config.Text.fontSize = canvas.height * 0.05;


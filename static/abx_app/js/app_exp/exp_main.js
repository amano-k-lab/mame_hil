/*******************************
Exp hil
*******************************/

//p5.js preload images
function preload() {

    //load images
    let num_imgs = exp_config.Condition.num_show_imgs;
    for (let i=0; i < num_imgs; ++i) {
    Config.Image.arrays_orig[i] = loadImage(STATIC_ROOT +IMAGE_ROOT+PATH_IMG_PRE+list_orig[exp_config.Condition.indices_target[i]]+'?23v'+APP_VER);
    Config.Image.arrays_fake[i] = loadImage(STATIC_ROOT +IMAGE_ROOT+PATH_IMG_PRE+list_fake[exp_config.Condition.indices_target[i]]+'?23v'+APP_VER);
  }
    Config.Image.num_imgs = num_imgs;
}

//p5.js initializing.
function setup() {
    let val_framerate= 60;

    createCanvas(canvas.width, canvas.height);
    angleMode(DEGREES);
    frameRate(val_framerate);

    exp_sceneManager = new exp_SceneManager();
    noCursor();
}

//p5.js frame animation.
function draw() {
    //Main experiment schedule
    exp_sceneManager.update();
}


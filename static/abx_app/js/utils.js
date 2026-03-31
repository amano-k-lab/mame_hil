// POST dictionary-style data from JavaScript to Django
function post(path, params, method='post') {
    // Function to ask for parameters of new episode
    // first create a hidden form:
    let form = document.getElementById('request');
    form.method = method;
    form.action = path;
    // Pass all parameters needed:
    for (const key in params) {
        if (params.hasOwnProperty(key)) {
            const hiddenField = document.createElement('input');
            hiddenField.type = 'hidden';
            hiddenField.name = key;
            hiddenField.value = params[key];
            form.appendChild(hiddenField)}
    }
    document.body.appendChild(form);
    form.submit();
}


// Screen size information
class CanvasInfo{
    constructor(width, height) {

        /*
        // For iPhone or Android
        if (navigator.userAgent.match(/iPhone|Android.+Mobile/)) {
            this.width = Math.max(window.outerWidth, window.outerHeight);
            this.height = Math.min(window.outerWidth, window.outerHeight);
        }
        // For PC
        else {
            this.width = Math.max(window.screen.availWidth, window.screen.availHeight);
            this.height = Math.min(window.screen.availWidth, window.screen.availHeight);
        }
        */

        //this.width = Math.max(window.innerWidth, window.innerHeight);
        //this.height = Math.min(window.innerWidth, window.innerHeight);
        this.width = width;
        this.height = height;

        this.center_x = this.width/2;
        this.center_y = this.height/2;
        this.left = 0;
        this.top = 0;
        this.right = this.width;
        this.bottom = this.height;
    }
}


// Utility functions for scenes
Scene_util = {

    // Whether the device is iOS
    isiOS: function () {
        return (typeof (DeviceOrientationEvent) !== 'undefined' && typeof (DeviceOrientationEvent.requestPermission) === 'function');
    },

    // Draw the background color
    drawBackGround: function(Scene, sceneManager){
        let bg_color;

        // Use the app main color for the start, instruction, and result scenes
        if([Scene.Start, Scene.Instruction, Scene.End].includes(sceneManager.scene)){
            bg_color = Config.BackGround.main_color;
        }
        // Use the stimulus background color during the task
        else{
            bg_color = Config.BackGround.task_color;
        }
        background(bg_color);
    },

    // Draw the fixation point
    drawFixation: function () {
        let cfg = Config.Fixation;
        let offset = cfg.d1 * 0.1;

        push();
        noStroke();
        fill(cfg.colorOval);
        ellipse(canvas.center_x, canvas.center_y, cfg.d1, cfg.d1);
        fill(cfg.colorCross);
        rect(canvas.center_x - (cfg.d2 / 2), canvas.center_y - (cfg.d1 / 2) - offset, cfg.d2, cfg.d1 + offset*2);
        rect(canvas.center_x - (cfg.d1 / 2) - offset, canvas.center_y - (cfg.d2 / 2), cfg.d1 + offset*2, cfg.d2);
        fill(cfg.colorOval);
        ellipse(canvas.center_x, canvas.center_y, cfg.d2, cfg.d2);
        pop();
    },

    // Draw text
    drawText: function(txt, pos_x, pos_y, fontSize = Config.Text.fontSize) {
        let cfg = Config.Text;
        push();
        fill(cfg.color);
        textSize(fontSize);
        textAlign(CENTER);
        text(txt, pos_x, pos_y);
        pop();
    },

    drawInfo(){
        push();
        strokeWeight(4);
        stroke(255, 204, 0);
        rect(canvas.right - (canvas.width * 0.2), canvas.top + (canvas.height * 0.02), canvas.width * 0.18,canvas.height * 0.15);
        fill(color(0, 0, 255));

        textSize(120);
        text('aaaaaaa', canvas.right - (canvas.width * 0.2), canvas.top + (canvas.height * 0.02), canvas.width * 0.18,canvas.height * 0.15);
        pop();
    },

    // Create a button
    createButton: function (text, size, position){
        let cfg = Config.Button;

        let button = createButton(text);
        button.style('font-size', cfg.text_size + 'px');
        button.size(size[0], size[1]);
        button.position(position[0], position[1]);

        button.show();

        return button;
    },

    // Get the current time as a string for database storage
    getCurrentTime: function(){
        return new Date().toLocaleString();
    }
};

const shuffle = ([...array]) => {
  for (let i = array.length - 1; i >= 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

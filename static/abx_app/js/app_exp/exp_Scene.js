
let exp_Scene = {};

exp_Scene.pre_define_stim = new Material();

//// Instruction scene
exp_Scene.Start = {

    init: function () {

    },

    update: function () {

        if (FLAG_READY=='ready'){
        Scene_util.drawText("Ready?", document.body.clientWidth/2, document.body.clientHeight/2);

        if (mouseIsPressed) {
            this.end();
        }

        // Allow progressing with the Enter key as well
        if(keyIsDown(ENTER)) {
            this.end();
        }
        } else if (FLAG_READY=='eye') {
            Scene_util.drawText("Please fixate on the center during stimulus presentation", document.body.clientWidth / 2, document.body.clientHeight / 2);

            if (mouseIsPressed) {
                this.end();
            }

            // Allow progressing with the Enter key as well
            if (keyIsDown(ENTER)) {
                this.end();
            }
        } else {
            this.end();
        }
    },

    end: function (){
        // Do not enter fullscreen. A browser popup appears, so maximize with F11 beforehand.
        //if (!Scene_util.isiOS()){
        //    fullscreen(true);
        //}
        exp_sceneManager.transition(exp_Scene.Stim_Blank_1)
    }
};



//// Blank scene 1
exp_Scene.Stim_Blank_1 = {
    init: function (){
        exp_Scene.pre_define_stim.define_condition(exp_sceneManager.list_answer[0], exp_sceneManager.firt_stim[0])
        this.startTime = Date.now();
    },


    update: function (){

        Scene_util.drawFixation();

        // End after a fixed amount of time
        let elapsedTime = (Date.now() - this.startTime) * 0.001;
        if (elapsedTime > exp_config.Condition.timeBlank_1) {
            this.end();
        }
    },


    end: function () {
            exp_sceneManager.transition(exp_Scene.Stim_First);
    },

};



//// Stimulus presentation scene 1
exp_Scene.Stim_First = {

    init: function (){
        this.startTime = Date.now();
        exp_sceneManager.get_eyetimestart();
    },


    update: function (){

        // Draw the stimulus
        exp_Scene.pre_define_stim.draw_1(exp_sceneManager.trial_count);
        Scene_util.drawFixation();

        // End after a fixed amount of time
       let elapsedTime = (Date.now() - this.startTime) * 0.001;
        if (elapsedTime > exp_config.Condition.timePresentation_1) {
            this.end();
        }
    },


    end: function () {
        exp_sceneManager.transition(exp_Scene.Stim_Blank_2);
    },

};


//// Blank scene 2
exp_Scene.Stim_Blank_2 = {

    isFirstBlank: true,

    init: function (){
        this.startTime = Date.now();
    },


    update: function (){

        Scene_util.drawFixation();

        // End after a fixed amount of time
        let elapsedTime = (Date.now() - this.startTime) * 0.001;
        if (elapsedTime > exp_config.Condition.timeBlank_2) {
            this.end();
        }
    },


    end: function () {
            exp_sceneManager.transition(exp_Scene.Stim_Second);
    },

};

//// Stimulus presentation scene 2
exp_Scene.Stim_Second = {

    init: function (){
        this.startTime = Date.now();
    },


    update: function (){

        // Draw the stimulus
        exp_Scene.pre_define_stim.draw_2(exp_sceneManager.trial_count);
        Scene_util.drawFixation();

        // End after a fixed amount of time
       let elapsedTime = (Date.now() - this.startTime) * 0.001;
        if (elapsedTime > exp_config.Condition.timePresentation_2) {
            this.end();
        }
    },


    end: function () {
        exp_sceneManager.transition(exp_Scene.Stim_Blank_3);
    },

};


//// Blank scene 3
exp_Scene.Stim_Blank_3 = {

    isFirstBlank: true,

    init: function (){
        this.startTime = Date.now();
    },


    update: function (){

        Scene_util.drawFixation();

        // End after a fixed amount of time
        let elapsedTime = (Date.now() - this.startTime) * 0.001;
        if (elapsedTime > exp_config.Condition.timeBlank_3) {
            this.end();
        }
    },


    end: function () {
            exp_sceneManager.transition(exp_Scene.Stim_Third);
    },

};


//// Stimulus presentation scene 3
exp_Scene.Stim_Third = {

    init: function (){
        this.startTime = Date.now();
    },


    update: function (){

        // Draw the stimulus
        exp_Scene.pre_define_stim.draw_3(exp_sceneManager.trial_count);
        Scene_util.drawFixation();

        // End after a fixed amount of time
       let elapsedTime = (Date.now() - this.startTime) * 0.001;
        if (elapsedTime > exp_config.Condition.timePresentation_3) {
            this.end();
        }
    },


    end: function () {
        exp_sceneManager.transition(exp_Scene.Stim_Blank_4);
    },

};


//// Blank scene 4
exp_Scene.Stim_Blank_4 = {

    isFirstBlank: true,

    init: function (){
        this.startTime = Date.now();
        exp_sceneManager.get_eyetimeend();
    },


    update: function (){

        Scene_util.drawFixation();

        // End after a fixed amount of time
        let elapsedTime = (Date.now() - this.startTime) * 0.001;
        if (elapsedTime > exp_config.Condition.timeBlank_4) {
            this.end();
        }
    },


    end: function () {
            exp_sceneManager.transition(exp_Scene.Stim_Response);
    },

};

//// Response screen scene
exp_Scene.Stim_Response = {

    init: function (){

        // Create response buttons
        let cfg = Config.Button;
        this.button_first = Scene_util.createButton('First', [cfg.width, cfg.height], [cfg.position[0]-cfg.space_offset, cfg.position[1]]);
        this.button_second = Scene_util.createButton('Second', [cfg.width, cfg.height], [cfg.position[0]+cfg.space_offset, cfg.position[1]]);

        // Set event handlers for response buttons
        this.button_first.mousePressed(() => this.onButtonPressed_first());
        this.button_second.mousePressed(() => this.onButtonPressed_second());

        this.startTime = Date.now();
        this.rt = 0;
    },


    update: function (){

        // Draw the fixation point
        Scene_util.drawFixation();

        // Display the prompt
        Scene_util.drawText('The third image was the same as...', canvas.center_x, canvas.bottom - (canvas.height * 0.25));

        // Display the progress
        //let txt = exp_sceneManager.trial_count+1 + '/' + exp_config.Condition.trialNum;
        //Scene_util.drawText(txt, canvas.center_x, canvas.top + (canvas.height * 0.4));

        // Allow responses with the arrow keys as well
        if(keyIsDown(LEFT_ARROW)) {
            this.onButtonPressed_first();
        }
        else if(keyIsDown(RIGHT_ARROW)){
            this.onButtonPressed_second();
        }

        this.rt = (Date.now() - this.startTime);
    },


    // When "First" is pressed
    onButtonPressed_first: function () {
        let hit;
        // Save the trial result
        if (exp_sceneManager.list_answer[0]=='first'){
            hit = 1;
        } else {
            hit = 0;
        }
        exp_sceneManager.updateResults('first', this.rt, hit);

        this.end();
    },


    // When "Second" is pressed
    onButtonPressed_second: function () {
        // Save the trial result
        let hit;
        if (exp_sceneManager.list_answer[0]=='second'){
            hit = 1;
        } else {
            hit = 0;
        }
        exp_sceneManager.updateResults('second', this.rt, hit);

        this.end();
    },


    end: function () {
        this.button_first.remove();
        this.button_second.remove();

        // Repeat the block until the specified block measurement is complete
        if (exp_sceneManager.trial_count <　exp_config.Condition.num_show_imgs){
            exp_sceneManager.transition(exp_Scene.Stim_Blank_1);
        }
        else{
            //exp_sceneManager.transition(exp_Scene.End);
            exp_sceneManager.quitTask();
        }
    },

};



//// Scene for moving to the next screen
 /*
exp_Scene.End = {

    init: function (){

        // Create the end button
        let cfg = Config.Button;
        this.button_end = Scene_util.createButton('END', [cfg.width, cfg.height], cfg.position);

        // Set the event handler for the end button
        this.button_end.mousePressed(() => this.onButtonPressed_end());
    },

    update: function (){
        let text_end = 'This session is finished.';
        Scene_util.drawText(text_end, canvas.center_x, canvas.center_y);
    },

    onButtonPressed_end: function(){
        this.end();
    },

    end: function () {
        this.button_end.remove();


        if (!Scene_util.isiOS()){
            fullscreen(false);
        }

        exp_sceneManager.quitTask();
    },
};
        */

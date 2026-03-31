class exp_SceneManager {

    scene = null;

    // Results
    results = {
        userName: userName,                                    // User name
        startDatetime: [],                                      // Measurement start time
        endDatetime: [],                                        // Measurement end time
        gazeStartTime: [],                                      // Gaze start time
        gazeEndTime: [],                                        // Gaze end time
        eachTrialNum: exp_config.Condition.num_show_imgs,          // Number of trials
        indices_target: [],                                 // Order of presented target indices
        trial_count: [],                                        // Trial count
        trial_response: [],                                     // Response for each trial
        trial_rt: [],                                           // Reaction time for each trial
        answer_target: [],                                      // Whether the answer was the first or second image
        first_stim: [],                                      // Whether the first stimulus was original or fake
        answer_hit: [],                                      // Whether the participant answered correctly
        average_hit: [],
        monitor_size: monitor_size,
    }

    list_trials = [];
    trial_count = 0;
    list_answer = [];

    constructor() {
        // Get the task start time
        this.results.startDatetime = Scene_util.getCurrentTime();

        // Initialize the stimulus conditions
        this.setupCondition();

        // Transition to the instruction scene
        this.transition(exp_Scene.Start);
    }

    // Update the scene, called every frame
    update() {
        // Draw the background color
        Scene_util.drawBackGround(exp_Scene, exp_sceneManager);
        // Update the scene
        this.scene.update()
    }


    // Switch scenes
    transition(scene) {
        this.scene = scene;
        this.scene.init();
    }


    // Set up conditions at the start of the experiment
    setupCondition() {
        let cfg = exp_config.Condition;
        this.results.indices_target = cfg.indices_target;
        this.list_answer = cfg.list_answer;
        this.list_answer = this.shuffleList(this.list_answer);
        this.firt_stim = cfg.first_stim;
        this.firt_stim = this.shuffleList(this.firt_stim);


    }

    // Update the result for each trial
    updateResults(resp, rt, hit) {
        this.results.trial_response.push(resp);
        this.results.trial_count.push(this.trial_count);
        this.results.trial_rt.push(rt);
        this.results.answer_target.push(this.list_answer[0]);
        this.results.first_stim.push(this.firt_stim[0]);
        this.results.answer_hit.push(hit)


        this.trial_count++;
        this.list_answer = this.shuffleList(this.list_answer);
        this.firt_stim = this.shuffleList(this.firt_stim);
    }

    // Shuffle the list
    shuffleList(list) {
        for (let i = list.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [list[i], list[j]] = [list[j], list[i]];
        }
        return list;
    }

    // Get timestamps related to eye movement
    get_eyetimestart() {
        this.results.gazeStartTime = Date.now();
    }

    get_eyetimeend() {
        this.results.gazeEndTime = Date.now();
    }


    quitTask() {
        // Get the task end time
        this.results.endDatetime = Scene_util.getCurrentTime();

        // Calculate the average accuracy
        let sum = this.results.answer_hit.reduce((acc, currentValue) => acc + currentValue, 0);
        let average = sum / this.results.answer_hit.length;

        // Convert lists to strings for saving
        this.results.answer_hit = JSON.stringify(this.results.answer_hit);
        this.results.answer_target = JSON.stringify(this.results.answer_target);
        this.results.trial_count = JSON.stringify(this.results.trial_count);
        this.results.trial_response = JSON.stringify(this.results.trial_response);
        this.results.trial_rt = JSON.stringify(this.results.trial_rt);
        this.results.first_stim = JSON.stringify(this.results.first_stim);
        this.results.average_hit = JSON.stringify(average);

        this.results.monitor_size = JSON.stringify(monitor_size);

        post(APP_ROOT + 'exp_material_end_view/', this.results, 'post');
    }

}

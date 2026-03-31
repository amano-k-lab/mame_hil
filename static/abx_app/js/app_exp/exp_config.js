
let exp_config = {};
// Experimental Conditions
exp_config.Condition = {};
exp_config.Condition.timePresentation_1 = 0.2; // Stimulus presentation time A
exp_config.Condition.timePresentation_2 = 0.2; // Stimulus presentation time B
exp_config.Condition.timePresentation_3 = 0.2; // Stimulus presentation time X
exp_config.Condition.timeBlank_1 = 1.5; // blank time in second
exp_config.Condition.timeBlank_2 = 0.5; // blank time in second
exp_config.Condition.timeBlank_3 = 0.5; // blank time in second
exp_config.Condition.timeBlank_4 = 1.0 // blank time in second
exp_config.Condition.num_show_imgs = 1;
exp_config.Condition.list_answer = ['first','second'];
exp_config.Condition.first_stim = ['orig','fake'];

//need to decide these indices to load images, not SceneMan ager
exp_config.Condition.indices_target = Array.from({length: exp_config.Condition.num_show_imgs}, (_, index) => index);

exp_config.Condition.indices_target = shuffle(exp_config.Condition.indices_target);

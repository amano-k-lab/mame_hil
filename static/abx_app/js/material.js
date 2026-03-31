
class Material {
    size_rescale;      // actual size
    startTime;      // Stimulus presentation start time


    constructor() {
        let cfg = Config.Image;
        this.arrays_orig = cfg.arrays_orig;
        this.arrays_fake = cfg.arrays_fake;
        this.size_rescale = cfg.size_rescale;
        this.startTime = Date.now();
        this.pos_target = [(canvas.center_x)-(this.size_rescale/2)+cfg.shift_right,(canvas.center_y)-(this.size_rescale/2)];
        this.array_image_1 = [];
        this.array_image_2 = [];

    }

    define_condition(cond,first_stim) {
        if (cond=='first' && first_stim=='orig'){
            this.array_image_1 = this.arrays_orig;
            this.array_image_2 = this.arrays_fake;
            this.array_image_3 = this.arrays_orig;
        } else if (cond=='second' && first_stim=='orig') {
            this.array_image_1 = this.arrays_orig;
            this.array_image_2 = this.arrays_fake;
            this.array_image_3 = this.arrays_fake;
        } else if (cond=='first' && first_stim=='fake') {
            this.array_image_1 = this.arrays_fake;
            this.array_image_2 = this.arrays_orig;
            this.array_image_3 = this.arrays_fake;
        } else if (cond=='second' && first_stim=='fake') {
            this.array_image_1 = this.arrays_fake;
            this.array_image_2 = this.arrays_orig;
            this.array_image_3 = this.arrays_orig;
        }
    }

    // Draw the stimulus
    draw_1(ind_trial){
        push();
        noStroke();
        // draw target image one
        image(this.array_image_1[ind_trial],this.pos_target[0],this.pos_target[1],this.size_rescale, this.size_rescale);

        pop();
    }

    draw_2(ind_trial){
        push();
        noStroke();
        // draw target image one
        image(this.array_image_2[ind_trial],this.pos_target[0],this.pos_target[1],this.size_rescale, this.size_rescale);
        pop();
    }

    draw_3(ind_trial){
        push();
        noStroke();
        // draw target image one
        image(this.array_image_3[ind_trial],this.pos_target[0],this.pos_target[1],this.size_rescale, this.size_rescale);
        pop();
    }
}


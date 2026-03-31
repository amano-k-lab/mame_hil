
class intro_SceneManager{

    scene = null;

    constructor() {

        // Transition to the instruction scene
        this.transition(intro_Scene.Instruction);
    }

    // Update the scene, called every frame
    update(){
        // Draw the background color
        Scene_util.drawBackGround(intro_Scene, intro_sceneManager);
        // Update the scene
        this.scene.update()
    }


    // Switch scenes
    transition(scene){
        this.scene = scene;
        this.scene.init();
    }

    quitTask(){

        location.href = APP_ROOT + 'introduction_end/';
    }

}

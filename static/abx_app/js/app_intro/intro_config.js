
let demo_Config = {};
// Slider
demo_Config.Slider = {};
demo_Config.Slider.width = Math.round(0.5 * canvas.height);
demo_Config.Slider.height = Math.round(0.02 * canvas.height);
demo_Config.Slider.position = [-demo_Config.Slider.width/2, canvas.center_y - (demo_Config.Slider.height / 2)];

// Slider value text
demo_Config.ValueText = {};
demo_Config.ValueText.color = 255;
demo_Config.ValueText.fontSize = canvas.height * 0.03;
demo_Config.ValueText.position = [canvas.center_x, Math.round(canvas.center_y + demo_Config.Slider.width*0.65)];

// Confirm button
demo_Config.DecideButton = {};
demo_Config.DecideButton.width = Config.Button.width;
demo_Config.DecideButton.height = Config.Button.height;
demo_Config.DecideButton.position = [Math.round(canvas.center_x + canvas.height*0.6 - (demo_Config.DecideButton.width/2)), Math.round(canvas.bottom - (canvas.height * 0.1) - (demo_Config.DecideButton.height/2))];
demo_Config.DecideButton.text_size = Math.round(canvas.height * 0.02);

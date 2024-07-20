<div align="center">
  
  <h1><img src="images/icon.png" width="35"> splatviz</h1>

![GitHub top language](https://img.shields.io/github/languages/top/Florian-barthel/splatviz) ![GitHub Release](https://img.shields.io/github/v/release/Florian-Barthel/splatviz) ![GitHub last commit](https://img.shields.io/github/last-commit/Florian-Barthel/splatviz) ![Static Badge](https://img.shields.io/badge/Platform-Linux-green) ![Static Badge](https://img.shields.io/badge/Platform-Windows-green)

</div>

<div align="justify"> 
This interactive viewer allows you to display and edit 3D Gaussian Splatting scenes in real time. By using the python GUI 
library (<a href="https://pyimgui.readthedocs.io/en/latest/guide/first-steps.html">pyimgui</a>) we can directly manipulate
the Gaussian python object just before rendering it. This enables endless editing and visualization possibilities. 
You can also evaluate python expressions just after the 
rendering, allowing you to access and debug the 3D scene just in time. Additionally, you can visualize multiple scenes
at the same time by comparing them side by side or in a splitscreen view. Lastly, you can also save renderings, ply files
of the current scene and even videos of a 360° rotation around the scene.

See some example editings in this  <a href="https://www.youtube.com/watch?v=AJGSa_wnZS8&ab_channel=FlorianBarthel">video</a>.
</div>
<img src="images/teaser.png">

## Install
Clone repository **recursively** in order to install glm from the diff_gaussian_rasterization package.
```
git clone https://github.com/Florian-Barthel/splatviz.git --recursive
```

Create conda environment with **conda**:

```bash
conda env create -f environment.yml
conda activate gs-view
```
Alternatively, create the environment with **micromamba**:
```bash
micromamba env create --file environment.yml --channel-priority flexible -y
micromamba activate gs-view
```

Launch:
`python run_main.py`

Launch with a specified path to look for .ply or .yml files:
`python run_main.py --data_path=path/with/plys`

## Widgets

<div align="justify"> 
  
### Load Widget
Once you run the `run_main.py` file, the viewer will directly load the first scene that is found in the data
directory you have specified. You can change the scene by clicking the **Browse** button. You will be displayed a list
of available _.ply_ files (or _.yml_ files that correspond to compressed gaussian scenes using 
<a href="https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/">this</a> compression method). If you have a lot of
scenes to chose from, you can also use the **Filter** textfield providing comma separated keywords (eg. _lego,30000,baseline_).
You will then only see those scenes that contain all keywords.

Additionally, you can also view multiple 3D scenes at once. Simply click **Add Scene**, which loads the same scene as 
before and then change the respective scene to another .ply file using the new browse button. The scenes are either 
rendered next to each other or in a split screen mode when activating the **Splitscreen** checkbox.

You can download some example scenes [here](https://drive.google.com/drive/folders/1RNMis5ibSrl2a87ZX614ZPsHKaY6ams6?usp=sharing).
They were created using the gaussian splatting compression method from: https://fraunhoferhhi.github.io/Self-Organizing-Gaussians.

<img src="images/load.png">
<hr>

### Edit Widget
The edit widget is the core functionality of this 3D viewer. It allows for real time editing of the gaussian 
python object during runtime. The code that you type in this text area will be executed just before the gaussian object is
forwarded to the cuda renderer. This means, the editing capabilities are unlimited. As long as the provided code
is executable, you can type any kind of python code and also import new libraries. An example could look like this, where
all scales are set to -8 (before activating with _exp()_) and all opacity values are set to 10 (before activating with
_sigmoid()_), while also the background is set to 1, which corresponds to white.

```python
gaussian._scaling = gaussian._scaling * 0 - 8
gaussian._opacity = gaussian._opacity * 0 + 10
self.bg_color[:] = 1
```

To enable smooth editing transitions, you can create sliders (press **Add Slider**) which you can access in the
editor text by typing slider.name (eg. slider.x). An example could look as follows. Here, we create a boolean mask
that filters all gaussians that are smaller than the value stored in **slider.x**.

```python
mask = torch.linalg.norm(gaussian._scaling, dim=-1) < slider.x

gaussian._xyz = gaussian._xyz[mask]
gaussian._rotation = gaussian._rotation[mask]
gaussian._scaling = gaussian._scaling[mask]
gaussian._opacity = gaussian._opacity[mask]
gaussian._features_dc = gaussian._features_dc[mask]
gaussian._features_rest = gaussian._features_rest[mask]
```

Lastly, you can save and load presets of code snippets so that you don't have to type the same code again after 
closing the application. Those presets are stored in a .json file (presets.json).

<img src="images/edit.png">
<hr>

### Eval Widget
The evaluate widget can be used to debug the gaussian splatting object. By typing
python code, which is executed after the rendering, you can access any variable 
from the rendering context and visualize them in a histogram. Some useful variables might be:
- gaussian
- render</li>   
- render_cam
- self

You can also access variables that you have defined in the editor of the Edit Widget.

<img src="images/eval.png">
<hr>

### Camera Widget
In the camera widget you can define the type and parameters of the camera. Most importantly, you can choose between the 
two modes **Orbit** and **WASD**.

In **Orbit** mode, the camera is looking at a specific point in 3D space, and you control the pitch and yaw of the camera 
rotating on a sphere around that point by dragging with the mouse over the rendered view.

In **WASD** mode, you fly through the scene using the mouse and the WASD keys similar to the camera controls in Unity.

**Important**: If the loaded scene is rotated incorrectly, you can adjust that by steering the camera so that it is 
looking straight up or down. Then set the up-vector to the current viewing direction by pressing **Set Current Direction**. 
If you were looking down, you will have to press the **Flip** button, since the scene is now upside down.

<img src="images/camera.png">
<hr>

### Video Widget
The video widget creates a video sequence of a full rotation around the current object.
Simply define the height of the camera and the rendering resolution. While the video is
rendering, the UI is frozen. A loading screen is shown in the terminal output.

🚧This feature is still under construction 🚧. Currently, it is not very intuitive to generate videos, as the camera
position is only defined by the <b>Camera Height</b> parameter. This will be changed in the future.

<img src="images/video.png">
<hr>

### Performance Widget
In the performance widget, you can track the FPS of the viewer and the FPS of the renderer. You can also specify 
limits for FPS and enable vertical synchronisation. In the future, there will also be a performance mode which 
skips some editing and evaluation calculations.

<img src="images/performance.png">
<hr>

### Render Widget
In the render tab, you can specify the rendering resolution (at the moment only squared resolutions) and the rendering mode.
Next to RGB, you can also render the image 3D depth and the alpha values.

<img src="images/render.png">
<hr>

### Save Widget
To save the .ply file or a rendering of the current (edited) 3D scene press the respective button in the 
save widget. Those files will be saved under _screenshots or _ply_files. The viewer might be frozen for a short 
while, after saving a ply file.

<img src="images/save.png">
</div>

## Future Plans
- Attach to Gaussian Splatting training for live tracking
- Test other 3DGS render for OSX support

## Recent Updates ⚡ 
- New Design
- Compare two or more Gaussian Splatting scenes side by side
- Improved camera controls
  - Orbit Mode: Camera moves on a sphere around a look_at_point
  - Free mode: Camera moves free and can be controlled with WASD

## Contribute
You are more than welcome to add further functionality or a better design to this interactive viewer!
The main goal is to create an easy-to-use tool that can be applied for debugging and for understanding
3D Gaussian Splatting objects. 


## Citation
If you find this viewer useful, please consider citing our work:
```
@misc{barthel2024gaussian,
    title={Gaussian Splatting Decoder for 3D-aware Generative Adversarial Networks}, 
    author={Florian Barthel and Arian Beckmann and Wieland Morgenstern and Anna Hilsmann and Peter Eisert},
    year={2024},
    eprint={2404.10625},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## References
This viewer is inspired by the visualizer from Efficient Geometry-aware 3D Generative Adversarial 
Networks (EG3D).

GUI: <a href="https://pyimgui.readthedocs.io/en/latest/guide/first-steps.html">pyimgui</a> and 
<a href="https://github.com/pthom/imgui_bundle">imgui_bundle</a> which are python wrappers for the c++ library 
<a href="https://github.com/ocornut/imgui">ImGUI</a>

Original code base: <a href="https://github.com/NVlabs/eg3d">EG3D</a>

3DGS: <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"> 3D Gaussian Splatting</a>

Compressing 3DGS scenes: <a href="https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/">Compact 3D Scene Representation via Self-Organizing Gaussian Grids</a>

3DGS Rasterizer with depth and alpha: <a href="https://github.com/slothfulxtx/diff-gaussian-rasterization">Diff rasterizer with depth and alpha</a>

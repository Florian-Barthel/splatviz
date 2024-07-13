# <img src="images/icon.png" width="30"> splatviz
This interactive viewer allows you to display and edit 3D Gaussian Splatting scenes in real time. By directly using a 
native python GUI library (imgui) we can directly manipulate the Gaussian python object just before the rendering it.
This enables endless editing and visualization possibilities. You can also evaluate python expression just after the 
rendering, allowing you to access and debug the 3D scene just in time. Additionally, you can visualize multiple scenes
at the same time comparing them side by side or in a splitscreen view. Lastly, you can also save renderings, ply files
of the current scene and even videos of a 360° rotation around the scene.

<img src="images/teaser.png">

## Install
Clone repository **recursively** in order to install glm from the diff_gaussian_rasterization package.
```
git clone https://github.com/Florian-Barthel/splatviz.git --recursive
```

Create conda environment:

```
conda env create -f environment.yml
conda activate gs-view
```

Launch:
`
python run_main.py
`

Launch with a specified path to look for .ply or .yml files:
`
python run_main.py --data_path=path/to/directory/with/gaussian/objects
`

## Widgets



<table>
  <tr>
    <td style="vertical-align: top; width: 38.2%">
      <h3>Load Widget</h3>
      <p>
      Once you run the run_main.py file, the viewer will directly load the first scene that is found in the data
      directory you have specified. You can change the scene by clicking the <b>Browse</b>> button. You will be displayed a list
      of available .ply files or .yml files that correspond to compressed gaussian scenes using 
      <a href="https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/">this</a> compression method. Additionally, you can also view
      multiple 3D scenes at once. Simply click <b>Add Scene</b>, which loads the same scene as before and then change the scene.
      The scenes are either rendered next to each other or in a split screen mode when activating the <b>Splitscreen</b> checkbox.
      </p>
    </td>
    <td>
      <img src="images/load.png">
    </td>
  </tr>

  <tr>
    <td style="vertical-align: top;">
      <h3>Edit Widget</h3>
      <p>
      The edit widget is the core functionality of this 3D viewer. It allows for real time editing of the gaussian 
      python object during runtime. The code that you type in this text area executed just before the gaussian object is
      forwarded to the cuda renderer. This means, the editing capabilities are unlimited. As long as the provided code
      is executable, you can type any kind of python code.<br>
      To enable smooth editing transitions, you can create sliders (press <b>Add Slider</b>) which you can access in the
      editor text by typing slider.name (eg. slider.x).<br>
      Lastly, you can save and load presets of code snippets so that you don't have to type the same code again after 
      closing the application. Those presets are stored in a .json file (presets.json).
      </p>
    </td>
    <td>
      <img src="images/edit.png">
    </td>
  </tr>

  <tr>
    <td style="vertical-align: top;">
      <h3>Eval Widget</h3>
      <p>
      The evaluate widget can be used to debug the gaussian splatting object. By typing
      python code, which is executed after the rendering, you can access any variable 
      from the rendering context and visualize them in a histogram. Some useful variables might be:
      <ul>
      <li>gaussian</li>      
      <li>render</li>   
      <li>render_cam</li>
      </ul>
      You can also access variables that you have defined in the editor of the Edit Widget.
      </p>
    </td>
    <td>
      <img src="images/eval_new.png">
    </td>
  </tr>

  <tr>
    <td style="vertical-align: top;">
      <h3>Camera Widget</h3>
      <p>
      In the camera widget you can define the type and parameters of the camera. Here, you can choose between the modes
      <b>Orbit</b> and <b>WASD</b>.<br>
      In the orbit mode, the camera is looking at a specific point in the 3D space, and you can control the pitch and yaw.
      <b>If the scene is rotated</b>, you can correct that by steering the camera so that it is looking straight up or down.
      Then set the up-vector to the current viewing direction by pressing <b>Set Current Direction</b>. You might have to press
      the <b>Flip</b> button if the scene is now upside down.<br>
      In the WASD mode, you can fly through the scene using the mouse and the WASD / arrow keys.<br>
      Generally, you can control the camera either by dragging the mouse over the rendered image or by using the WASD or
      the arrow keys.
      </p>
    </td>
    <td style="text-align: center;">
      <img src="images/camera.png">
    </td>
  </tr>

  <tr>
    <td style="vertical-align: top;">
      <h3>Video Widget</h3>
      <p>
      The video widget creates a video sequence of a full rotation around the current object.
      Simply define the height of the camera and the rendering resolution. While the video is
      rendering, the UI is frozen. A loading screen is shown in the terminal output. <br>
      🚧This feature is still under construction 🚧. Currently, it is not very intuitive to generate videos, as the camera
      position is only defined by the <b>Camera Height</b> parameter. This will be changed in the future.
      </p>
    </td>
    <td style="text-align: center;">
      <img src="images/video.png">
    </td>
  </tr>

  <tr>
    <td style="vertical-align: top;">
      <h3>Performance Widget</h3>
      <p>
      In the performance widget, you can track the FPS of the viewer and the FPS of the renderer. You can also specify 
      limits for FPS and enable vertical synchronisation. In the future, there will also be a performance mode which 
      skips some editing and evaluation calculations.
      </p>
    </td>
    <td>
      <img src="images/performance.png">
    </td>
  </tr>

  <tr>
    <td style="vertical-align: top;">
      <h3>Render Widget</h3>
      <p>
      In the render tab, you can specify the rendering resolution (at the moment only squared resolutions) and the rendering mode.
      Next to RGB, you can also render the image 3D depth and the alpha values.
      </p>
    </td>
    <td>
      <img src="images/render.png">
    </td>
  </tr>

  <tr>
    <td style="vertical-align: top;">
      <h3>Save Widget</h3>
      <p>
      To save the .ply file or a rendering of the current (edited) 3D scene simply press the respective button in the 
      save widget. Those files will be saved under: _screenshots or _ply_files. The viewer might be frozen for a short 
      while, after saving a ply file.
      </p>
    </td>
    <td>
      <img src="images/save.png">
    </td>
  </tr>
</table>


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

GUI: <a href="https://pyimgui.readthedocs.io/en/latest/guide/first-steps.html">pyimgui</a> and <a href="https://github.com/pthom/imgui_bundle">imgui_bundle</a>

Original code base: <a href="https://github.com/NVlabs/eg3d">EG3D</a>

3DGS: <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"> 3D Gaussian Splatting</a>

Compressing 3DGS scenes: <a href="https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/">Compact 3D Scene Representation via Self-Organizing Gaussian Grids</a>

3DGS Rasterizer with depth and alpha: <a href="https://github.com/slothfulxtx/diff-gaussian-rasterization">Diff rasterizer with depth and alpha</a>

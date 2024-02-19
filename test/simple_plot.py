import numpy as np
from imgui_bundle._imgui_bundle import implot
import imgui

imgui.create_context()
implot.create_context()

imgui.new_frame()
imgui.begin("test", True)
if implot.begin_plot("hallo"):
    implot.plot_bars("bars", xs=np.arange(10), ys=np.arange(10), bar_size=2)
    implot.end_plot()
imgui.end()
imgui.render()

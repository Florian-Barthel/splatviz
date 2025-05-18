# -*- coding: utf-8 -*-
import os
import sys

# For Linux/Wayland users.
if os.getenv("XDG_SESSION_TYPE") == "wayland":
    os.environ["XDG_SESSION_TYPE"] = "x11"

import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer

active = {
    "window": True,
    "child": False,
    "tooltip": False,
    "menu bar": False,
    "popup": False,
    "popup modal": False,
    "popup context item": False,
    "popup context window": False,
    "drag drop": False,
    "group": False,
    "tab bar": False,
    "list box": False,
    "popup context void": False,
    "table": False,
}
path_to_font = None  # "path/to/font.ttf"

opened_state = True

# Frame commands from the video
# def frame_commands():
#     io = imgui.get_io()
#     if io.key_ctrl and io.keys_down[glfw.KEY_Q]:
#         sys.exit(0)
#
#     if imgui.begin_main_menu_bar():
#         if imgui.begin_menu("File"):
#             clicked, selected = imgui.menu_item("Quit", "Ctrl+Q")
#             if clicked:
#                 sys.exit(0)
#             imgui.end_menu()
#         imgui.end_main_menu_bar()
#
#     with imgui.begin("A Window!"):
#         if imgui.button("select"):
#             imgui.open_popup("select-popup")
#
#         try:
#             with imgui.begin_popup("select-popup") as popup:
#                 if popup.opened:
#                     imgui.text("Select one")
#                     raise Exception
#         except Exception:
#             print("caught exception and no crash!")


def frame_commands():
    io = imgui.get_io()

    if io.key_ctrl and io.keys_down[glfw.KEY_Q]:
        sys.exit(0)

    with imgui.begin_main_menu_bar() as main_menu_bar:
        if main_menu_bar.opened:
            with imgui.begin_menu("File", True) as file_menu:
                if file_menu.opened:
                    clicked_quit, selected_quit = imgui.menu_item("Quit", "Ctrl+Q")
                    if clicked_quit:
                        sys.exit(0)

    # turn examples on/off
    with imgui.begin("Active examples"):
        for label, enabled in active.copy().items():
            _, enabled = imgui.checkbox(label, enabled)
            active[label] = enabled

    if active["window"]:
        with imgui.begin("Hello, Imgui!"):
            imgui.text("Hello, World!")

    if active["child"]:
        with imgui.begin("Example: child region"):
            with imgui.begin_child("region", 150, -50, border=True):
                imgui.text("inside region")
            imgui.text("outside region")

    if active["tooltip"]:
        with imgui.begin("Example: tooltip"):
            imgui.button("Click me!")
            if imgui.is_item_hovered():
                with imgui.begin_tooltip():
                    imgui.text("This button is clickable.")

    if active["menu bar"]:
        try:
            flags = imgui.WINDOW_MENU_BAR
            with imgui.begin("Child Window - File Browser", flags=flags):
                with imgui.begin_menu_bar() as menu_bar:
                    if menu_bar.opened:
                        with imgui.begin_menu('File') as file_menu:
                            if file_menu.opened:
                                clicked, state = imgui.menu_item('Close')
                                if clicked:
                                    active["menu bar"] = False
                                    raise Exception
        except Exception:
            print("exception handled")

    if active["popup"]:
        with imgui.begin("Example: simple popup"):
            if imgui.button("select"):
                imgui.open_popup("select-popup")
            imgui.same_line()
            with imgui.begin_popup("select-popup") as popup:
                if popup.opened:
                    imgui.text("Select one")
                    imgui.separator()
                    imgui.selectable("One")
                    imgui.selectable("Two")
                    imgui.selectable("Three")

    if active["popup modal"]:
        with imgui.begin("Example: simple popup modal"):
            if imgui.button("Open Modal popup"):
                imgui.open_popup("select-popup-modal")
            imgui.same_line()
            with imgui.begin_popup_modal("select-popup-modal") as popup:
                if popup.opened:
                    imgui.text("Select an option:")
                    imgui.separator()
                    imgui.selectable("One")
                    imgui.selectable("Two")
                    imgui.selectable("Three")

    if active["popup context item"]:
        with imgui.begin("Example: popup context view"):
            imgui.text("Right-click to set value.")
            with imgui.begin_popup_context_item("Item Context Menu") as popup:
                if popup.opened:
                    imgui.selectable("Set to Zero")

    if active["popup context window"]:
        with imgui.begin("Example: popup context window"):
            with imgui.begin_popup_context_window() as popup:
                if popup.opened:
                    imgui.selectable("Clear")

    if active["popup context void"]:
        with imgui.begin_popup_context_void() as popup:
            if popup.opened:
                imgui.selectable("Clear")

    if active["drag drop"]:
        with imgui.begin("Example: drag and drop"):
            imgui.button('source')
            with imgui.begin_drag_drop_source() as src:
                if src.dragging:
                    imgui.set_drag_drop_payload('itemtype', b'payload')
                    imgui.button('dragged source')
            imgui.button('dest')
            with imgui.begin_drag_drop_target() as dst:
                if dst.hovered:
                    payload = imgui.accept_drag_drop_payload('itemtype')
                    if payload is not None:
                        print('Received:', payload)

    if active["group"]:
        with imgui.begin("Example: item groups"):
            with imgui.begin_group():
                imgui.text("First group (buttons):")
                imgui.button("Button A")
                imgui.button("Button B")
            imgui.same_line(spacing=50)
            with imgui.begin_group():
                imgui.text("Second group (text and bullet texts):")
                imgui.bullet_text("Bullet A")
                imgui.bullet_text("Bullet B")

    if active["tab bar"]:
        with imgui.begin("Example Tab Bar"):
            with imgui.begin_tab_bar("MyTabBar") as tab_bar:
                if tab_bar.opened:
                    with imgui.begin_tab_item("Item 1") as item1:
                        if item1.opened:
                            imgui.text("Here is the tab content!")
                    with imgui.begin_tab_item("Item 2") as item2:
                        if item2.opened:
                            imgui.text("Another content...")
                    global opened_state
                    with imgui.begin_tab_item("Item 3", opened=opened_state) as item3:
                        opened_state = item3.opened
                        if item3.selected:
                            imgui.text("Hello Saylor!")

    if active["list box"]:
        with imgui.begin("Example: custom listbox"):
            with imgui.begin_list_box("List", 200, 100) as list_box:
                if list_box.opened:
                    imgui.selectable("Selected", True)
                    imgui.selectable("Not Selected", False)

    if active["table"]:
        with imgui.begin("Example: table"):
            with imgui.begin_table("data", 2) as table:
                if table.opened:
                    imgui.table_next_column()
                    imgui.table_header("A")
                    imgui.table_next_column()
                    imgui.table_header("B")

                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text("123")

                    imgui.table_next_column()
                    imgui.text("456")

                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text("789")

                    imgui.table_next_column()
                    imgui.text("111")

                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text("222")

                    imgui.table_next_column()
                    imgui.text("333")


def render_frame(impl, window, font):
    glfw.poll_events()
    impl.process_inputs()
    imgui.new_frame()

    gl.glClearColor(0.1, 0.1, 0.1, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    if font is not None:
        imgui.push_font(font)
    frame_commands()
    if font is not None:
        imgui.pop_font()

    imgui.render()
    impl.render(imgui.get_draw_data())
    glfw.swap_buffers(window)


def impl_glfw_init():
    width, height = 1920, 1080
    window_name = "minimal ImGui/GLFW3 example"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window


def main():
    imgui.create_context()
    window = impl_glfw_init()

    impl = GlfwRenderer(window)

    io = imgui.get_io()
    jb = io.fonts.add_font_from_file_ttf(path_to_font, 30) if path_to_font is not None else None
    impl.refresh_font_texture()

    while not glfw.window_should_close(window):
        render_frame(impl, window, jb)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
from vispy import scene
canvas = scene.SceneCanvas(keys=None)
view = canvas.central_widget.add_view()
view.camera = "panzoom"
print(dir(view.camera.events))

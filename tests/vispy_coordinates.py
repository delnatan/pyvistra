import sys

import numpy as np
import vispy.scene
from vispy.visuals.transforms import STTransform

# 1. Setup Canvas
canvas = vispy.scene.SceneCanvas(
    keys="interactive", show=True, title="VisPy Coordinate Exploration"
)
view = canvas.central_widget.add_view()

# 2. Generate Data
image_data = np.random.randn(100, 100).astype(np.float32)

# 3. Setup Image
image = vispy.scene.visuals.Image(image_data, parent=view.scene, method="auto")
image.order = 0  # Draw first (background)

# Apply a generic scale just to prove 'Visual' coords differ from 'Scene' coords
# This scales the image data by 2x, so the image will appear 200 units wide in the Scene
transform = STTransform(scale=(1, 4))
image.transform = transform

# 4. Setup Marker (The Cursor)
marker = vispy.scene.visuals.Markers(parent=view.scene)
marker.order = 10  # Draw last (foreground)
marker.set_gl_state(
    "translucent", depth_test=False
)  # Disable depth check for visibility
marker.set_data(pos=np.array([[0, 0]]), face_color="red", size=15)

# 5. Setup Camera
view.camera = vispy.scene.PanZoomCamera(aspect=1)
# IMPORTANT: Frame the view around the image (approx 0 to 200 due to 2x scale)
view.camera.set_range(x=(-10, 210), y=(-10, 210))


def on_mouse_click(event):
    if "Shift" in event.modifiers:
        # --- A. Canvas -> Visual (Data Indices) ---
        # Maps mouse pixels directly to the image array indices (0-100)
        tr_to_data = image.get_transform(map_from="canvas", map_to="visual")
        visual_pos = tr_to_data.map(event.pos)

        # --- B. Canvas -> Scene (World Coordinates) ---
        # Maps mouse pixels to the ViewBox coordinate system
        # This is where the Camera and the Marker live.
        tr_to_scene = image.get_transform(map_from="canvas", map_to="scene")
        scene_pos = tr_to_scene.map(event.pos)

        # Normalize vectors (x, y, z, w) -> (x, y)
        data_x, data_y = visual_pos[0], visual_pos[1]
        scene_x, scene_y = scene_pos[0], scene_pos[1]

        print(f"1. Canvas (Screen Pixels): {event.pos}")
        print(f"2. Scene  (World/Camera):  ({scene_x:.2f}, {scene_y:.2f})")
        print(f"3. Visual (Image Data):    ({data_x:.2f}, {data_y:.2f})")

        # Validation:
        # Because we scaled the image by 2x:
        # Scene Coordinate should be exactly 2x the Visual Coordinate.
        print("-" * 50)

        # Update Marker Position (Markers live in Scene coordinates)
        marker.set_data(
            pos=np.array([[scene_x, scene_y]]), face_color="red", size=15
        )
        canvas.update()
    else:
        return


canvas.events.mouse_press.connect(on_mouse_click)

if __name__ == "__main__":
    if sys.flags.interactive != 1:
        vispy.app.run()

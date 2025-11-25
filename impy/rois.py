import numpy as np
from vispy import scene

class ROI:
    def __init__(self, view, name="ROI"):
        self.view = view
        self.name = name
        self.visuals = []
        self.data = {} # Store geometry data for serialization

    def set_visible(self, visible):
        for v in self.visuals:
            v.visible = visible

    def remove(self):
        for v in self.visuals:
            v.parent = None
        self.visuals = []
        
    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "data": self.data
        }
        
    def from_dict(self, data):
        self.data = data
        self._update_visuals_from_data()

    def _update_visuals_from_data(self):
        pass

class CoordinateROI(ROI):
    def __init__(self, view, name="Coordinate"):
        super().__init__(view, name)
        self.origin = None
        self.vector = None
        
        # Visuals
        self.line = scene.visuals.Line(
            pos=np.zeros((3, 2)), 
            color=["red", "green"], 
            width=2, 
            connect="segments",
            parent=self.view.scene
        )
        self.marker = scene.visuals.Markers(
            pos=np.zeros((1, 3)),
            face_color="yellow",
            size=8,
            parent=self.view.scene
        )
        # Arrowhead for primary vector
        self.arrow = scene.visuals.Arrow(
            pos=np.zeros((2, 3)),
            color="red",
            width=4,
            arrow_size=20,
            arrow_type="stealth",
            parent=self.view.scene
        )
        
        self.visuals.extend([self.line, self.marker, self.arrow])

    def update(self, p1, p2):
        """
        p1: Origin (x, y)
        p2: End of Anterior (Primary) vector (x, y)
        """
        self.origin = np.array(p1)
        anterior_vec = np.array(p2) - self.origin
        self.vector = anterior_vec # Keep for compatibility if needed, but prefer specific names
        
        # Orthogonal vector (Dorsal) (-y, x)
        dorsal_vec = np.array([-anterior_vec[1], anterior_vec[0]])
        
        # Calculate end points
        anterior_end = self.origin + anterior_vec
        dorsal_end = self.origin + dorsal_vec
        
        # Store data with new terminology
        self.data = {
            "origin": p1, 
            "anterior": tuple(anterior_end),
            "dorsal": tuple(dorsal_end)
        }
        
        # Dorsal Line (Green)
        dorsal_line_3d = np.zeros((2, 3))
        dorsal_line_3d[0, :2] = self.origin
        dorsal_line_3d[1, :2] = dorsal_end
        self.line.set_data(pos=dorsal_line_3d, color="green")
        
        # Anterior Arrow (Red)
        arrow_pos = np.zeros((2, 3))
        arrow_pos[0, :2] = self.origin
        arrow_pos[1, :2] = anterior_end
        self.arrow.set_data(pos=arrow_pos, color="red")
        
        marker_pos = np.zeros((1, 3))
        marker_pos[0, :2] = self.origin
        self.marker.set_data(pos=marker_pos)
        
    def _update_visuals_from_data(self):
        # Support old format ("end") and new format ("anterior")
        if "origin" in self.data:
            origin = self.data["origin"]
            if "anterior" in self.data:
                anterior = self.data["anterior"]
                # We can just call update, which recalculates dorsal. 
                # If we want to support custom dorsal in future, we'd need to change update signature
                # But for now, dorsal is strictly orthogonal to anterior in the drawing tool.
                # However, if loaded data has a specific dorsal, we might want to respect it?
                # The user said "include the direction or coordinate for the orthogonal vector when we save".
                # Implies we might want to read it back.
                # But `update` currently enforces orthogonality. 
                # Let's assume for now we just restore based on anterior to keep it simple and consistent with the tool behavior.
                self.update(origin, anterior)
            elif "end" in self.data:
                # Legacy support
                self.update(origin, self.data["end"])

class RectangleROI(ROI):
    def __init__(self, view, name="Rectangle"):
        super().__init__(view, name)
        self.rect = scene.visuals.Rectangle(
            center=(0, 0, 0), width=1, height=1,
            border_color="yellow", color=(1, 1, 0, 0.1),
            parent=self.view.scene
        )
        self.visuals.append(self.rect)

    def update(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        self.data = {"p1": p1, "p2": p2}
        
        # Rectangle center is center of box
        cx = x + w / 2
        cy = y + h / 2
        
        self.rect.center = (cx, cy, 0)
        self.rect.width = w
        self.rect.height = h
        
    def _update_visuals_from_data(self):
        if "p1" in self.data and "p2" in self.data:
            self.update(self.data["p1"], self.data["p2"])

class CircleROI(ROI):
    def __init__(self, view, name="Circle"):
        super().__init__(view, name)
        self.circle = scene.visuals.Ellipse(
            center=(0, 0, 0), radius=1,
            border_color="cyan", color=(0, 1, 1, 0.1),
            parent=self.view.scene
        )
        self.visuals.append(self.circle)

    def update(self, p1, p2):
        # p1 is center, p2 defines radius
        cx, cy = p1
        dx = p2[0] - cx
        dy = p2[1] - cy
        radius = np.sqrt(dx**2 + dy**2)
        
        self.data = {"center": p1, "edge": p2}
        
        self.circle.center = (cx, cy, 0)
        self.circle.radius = radius
        
    def _update_visuals_from_data(self):
        if "center" in self.data and "edge" in self.data:
            self.update(self.data["center"], self.data["edge"])

class LineROI(ROI):
    def __init__(self, view, name="Line"):
        super().__init__(view, name)
        self.line = scene.visuals.Line(
            pos=np.zeros((2, 3)),
            color="magenta",
            width=2,
            parent=self.view.scene
        )
        self.visuals.append(self.line)
        
    def update(self, p1, p2):
        self.data = {"p1": p1, "p2": p2}
        
        pos = np.zeros((2, 3))
        pos[0, :2] = p1
        pos[1, :2] = p2
        self.line.set_data(pos=pos)
        
    def _update_visuals_from_data(self):
        if "p1" in self.data and "p2" in self.data:
            self.update(self.data["p1"], self.data["p2"])

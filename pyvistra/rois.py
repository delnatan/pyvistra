import numpy as np
from vispy import scene

class ROI:
    def __init__(self, view, name="ROI"):
        self.view = view
        self.name = name
        self.visuals = []
        self.data = {} # Store geometry data for serialization
        
        # Editing State
        self.selected = False
        self.handle_visual = scene.visuals.Markers(
            parent=self.view.scene, 
            face_color='white', 
            edge_color='blue',
            size=12
        )
        self.handle_visual.visible = False
        self.visuals.append(self.handle_visual)
        self.handle_points = {} # id -> (x, y)

    def set_visible(self, visible):
        for v in self.visuals:
            # Don't show handles if not selected, even if ROI is visible
            if v is self.handle_visual:
                v.visible = visible and self.selected
            else:
                v.visible = visible

    def remove(self):
        for v in self.visuals:
            v.parent = None
        self.visuals = []
        
    def select(self, active):
        self.selected = active
        self.handle_visual.visible = active
        if active:
            self._update_handles()
            
    def _update_handles(self):
        """Update the positions of the handle visual based on current geometry."""
        pass

    def hit_test(self, point):
        """
        Return handle_id if hit, 'center' if body hit, or None.
        point: (x, y) in data coordinates.
        """
        # 1. Check handles
        if self.selected:
            for hid, pos in self.handle_points.items():
                dist = np.linalg.norm(np.array(point) - np.array(pos))
                # Threshold depends on zoom, but let's assume data coords for now.
                # Ideally we project to screen coords for hit testing, but we don't have easy access to transform here?
                # We can approximate.
                if dist < 5: # 5 units tolerance? Might be too small/large depending on image scale.
                    return hid
        return None

    def move(self, delta):
        """Move the entire ROI by delta (dx, dy)."""
        pass

    def adjust(self, handle_id, new_pos):
        """Move a specific handle to new_pos."""
        pass

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
        self.flipped = False
        
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
            symbol="x",
            edge_color="blue",
            edge_width=2,
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
        # If flipped, we negate it (or just rotate the other way)
        if self.flipped:
            dorsal_vec = np.array([anterior_vec[1], -anterior_vec[0]])
        else:
            dorsal_vec = np.array([-anterior_vec[1], anterior_vec[0]])
        
        # Calculate end points
        anterior_end = self.origin + anterior_vec
        dorsal_end = self.origin + dorsal_vec
        
        # Store data with new terminology
        self.data = {
            "origin": p1, 
            "anterior": tuple(anterior_end),
            "dorsal": tuple(dorsal_end),
            "flipped": self.flipped
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
        self.marker.set_data(
            pos=marker_pos,
            symbol="x",
            edge_color="blue",
            edge_width=2,
            size=8
        )
        
        if self.selected:
            self._update_handles()
        
    def _update_visuals_from_data(self):
        # Support old format ("end") and new format ("anterior")
        if "origin" in self.data:
            # Restore state
            self.flipped = self.data.get("flipped", False)
            
            origin = self.data["origin"]
            if "anterior" in self.data:
                anterior = self.data["anterior"]
                self.update(origin, anterior)
            elif "end" in self.data:
                # Legacy support
                self.update(origin, self.data["end"])

    def flip(self):
        """Flip the dorsal vector direction."""
        self.flipped = not self.flipped
        if "origin" in self.data and "anterior" in self.data:
            self.update(self.data["origin"], self.data["anterior"])

    def _update_handles(self):
        if "origin" not in self.data: return
        
        origin = self.data["origin"]
        anterior = self.data["anterior"] if "anterior" in self.data else self.data.get("end")
        
        self.handle_points = {
            "origin": origin,
            "anterior": anterior
        }
        
        pts = list(self.handle_points.values())
        self.handle_visual.set_data(pos=np.array(pts), face_color='white', size=10)

    def hit_test(self, point):
        # 1. Check handles
        hid = super().hit_test(point)
        if hid: return hid
        
        # 2. Check lines? For now just handles are enough for adjustment.
        # Maybe check proximity to the main line for moving?
        if "origin" in self.data and "anterior" in self.data:
            p1 = np.array(self.data["origin"])
            p2 = np.array(self.data["anterior"])
            p = np.array(point)
            
            # Distance from point to segment
            # Project p onto line p1-p2
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0: return None
            t = np.dot(p - p1, p2 - p1) / l2
            t = max(0, min(1, t))
            projection = p1 + t * (p2 - p1)
            dist = np.linalg.norm(p - projection)
            
            if dist < 5:
                return 'center'
                
        return None

    def move(self, delta):
        if "origin" in self.data:
            dx, dy = delta
            origin = self.data["origin"]
            anterior = self.data["anterior"]
            
            new_origin = (origin[0] + dx, origin[1] + dy)
            new_anterior = (anterior[0] + dx, anterior[1] + dy)
            self.update(new_origin, new_anterior)

    def adjust(self, handle_id, new_pos):
        if "origin" not in self.data: return
        
        origin = self.data["origin"]
        anterior = self.data["anterior"]
        
        if handle_id == "origin":
            self.update(new_pos, anterior)
        elif handle_id == "anterior":
            self.update(origin, new_pos)

class RectangleROI(ROI):
    def __init__(self, view, name="Rectangle"):
        super().__init__(view, name)
        self.rect = scene.visuals.Rectangle(
            center=(0, 0, 0), width=1, height=1,
            border_color="yellow", color=(1, 1, 0, 0.1),
            parent=self.view.scene
        )
        self.rect.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False
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
        
        # Ensure non-zero width/height to avoid Vispy errors
        w = max(w, 1e-6)
        h = max(h, 1e-6)
        
        self.rect.center = (cx, cy, 0)
        self.rect.width = w
        self.rect.height = h
        
        if self.selected:
            self._update_handles()
        
    def _update_handles(self):
        if "p1" not in self.data: return
        
        p1 = self.data["p1"]
        p2 = self.data["p2"]
        x1, y1 = p1
        x2, y2 = p2
        
        # Define 4 corners
        # We need to know which is which to keep p1/p2 logic consistent?
        # Actually, p1 and p2 are just diagonal corners.
        # Let's define handles for all 4 corners to allow free resizing.
        # But for simplicity, let's just show p1 and p2? 
        # No, users expect 4 corners.
        
        # Let's normalize
        l, r = min(x1, x2), max(x1, x2)
        t, b = min(y1, y2), max(y1, y2)
        
        self.handle_points = {
            "tl": (l, t),
            "tr": (r, t),
            "bl": (l, b),
            "br": (r, b)
        }
        
        pts = list(self.handle_points.values())
        self.handle_visual.set_data(pos=np.array(pts), face_color='white', size=10)

    def hit_test(self, point):
        # 1. Check handles
        hid = super().hit_test(point)
        if hid: return hid
        
        # 2. Check body (inside rect)
        if "p1" in self.data:
            p1 = self.data["p1"]
            p2 = self.data["p2"]
            x1, y1 = p1
            x2, y2 = p2
            l, r = min(x1, x2), max(x1, x2)
            t, b = min(y1, y2), max(y1, y2)
            
            px, py = point
            if l <= px <= r and t <= py <= b:
                return 'center'
                
        return None

    def move(self, delta):
        if "p1" in self.data:
            dx, dy = delta
            p1 = self.data["p1"]
            p2 = self.data["p2"]
            
            new_p1 = (p1[0] + dx, p1[1] + dy)
            new_p2 = (p2[0] + dx, p2[1] + dy)
            self.update(new_p1, new_p2)

    def adjust(self, handle_id, new_pos):
        # handle_id is tl, tr, bl, br
        # We need to update p1/p2 such that the rect matches the new corner
        # This implies p1/p2 might swap.
        
        if "p1" not in self.data: return
        
        # Current bounds
        p1 = self.data["p1"]
        p2 = self.data["p2"]
        l, r = min(p1[0], p2[0]), max(p1[0], p2[0])
        t, b = min(p1[1], p2[1]), max(p1[1], p2[1])
        
        nx, ny = new_pos
        
        if handle_id == "tl":
            l, t = nx, ny
        elif handle_id == "tr":
            r, t = nx, ny
        elif handle_id == "bl":
            l, b = nx, ny
        elif handle_id == "br":
            r, b = nx, ny
            
        # Reconstruct p1, p2
        self.update((l, t), (r, b))

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
        self.circle.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False
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
        self.circle.radius = max(radius, 1e-6)
        
        if self.selected:
            self._update_handles()
            
    def _update_handles(self):
        if "center" not in self.data: return
        
        center = self.data["center"]
        edge = self.data["edge"]
        
        self.handle_points = {
            "center": center,
            "edge": edge
        }
        
        pts = list(self.handle_points.values())
        self.handle_visual.set_data(pos=np.array(pts), face_color='white', size=10)

    def hit_test(self, point):
        # 1. Check handles
        hid = super().hit_test(point)
        if hid: return hid
        
        # 2. Check body (inside circle)
        if "center" in self.data:
            cx, cy = self.data["center"]
            px, py = point
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            if dist <= self.circle.radius:
                return 'center'
                
        return None

    def move(self, delta):
        if "center" in self.data:
            dx, dy = delta
            cx, cy = self.data["center"]
            ex, ey = self.data["edge"]
            
            new_center = (cx + dx, cy + dy)
            new_edge = (ex + dx, ey + dy)
            self.update(new_center, new_edge)

    def adjust(self, handle_id, new_pos):
        if "center" not in self.data: return
        
        center = self.data["center"]
        edge = self.data["edge"]
        
        if handle_id == "center":
            # Moving center moves the whole circle? Or just center (changing radius)?
            # Usually center handle moves the object. But we have 'move' for that.
            # If user drags center handle, they expect move.
            # But here adjust is called when dragging a handle.
            # Let's make center handle move the circle.
            dx = new_pos[0] - center[0]
            dy = new_pos[1] - center[1]
            self.move((dx, dy))
        elif handle_id == "edge":
            # Change radius
            self.update(center, new_pos)
        
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
        
        if self.selected:
            self._update_handles()
            
    def _update_handles(self):
        if "p1" not in self.data: return
        
        p1 = self.data["p1"]
        p2 = self.data["p2"]
        
        self.handle_points = {
            "p1": p1,
            "p2": p2
        }
        
        pts = list(self.handle_points.values())
        self.handle_visual.set_data(pos=np.array(pts), face_color='white', size=10)

    def hit_test(self, point):
        # 1. Check handles
        hid = super().hit_test(point)
        if hid: return hid
        
        # 2. Check proximity to line
        if "p1" in self.data:
            p1 = np.array(self.data["p1"])
            p2 = np.array(self.data["p2"])
            p = np.array(point)
            
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0: return None
            t = np.dot(p - p1, p2 - p1) / l2
            t = max(0, min(1, t))
            projection = p1 + t * (p2 - p1)
            dist = np.linalg.norm(p - projection)
            
            if dist < 5:
                return 'center'
                
        return None

    def move(self, delta):
        if "p1" in self.data:
            dx, dy = delta
            p1 = self.data["p1"]
            p2 = self.data["p2"]
            
            new_p1 = (p1[0] + dx, p1[1] + dy)
            new_p2 = (p2[0] + dx, p2[1] + dy)
            self.update(new_p1, new_p2)

    def adjust(self, handle_id, new_pos):
        if "p1" not in self.data: return
        
        p1 = self.data["p1"]
        p2 = self.data["p2"]
        
        if handle_id == "p1":
            self.update(new_pos, p2)
        elif handle_id == "p2":
            self.update(p1, new_pos)
        
    def _update_visuals_from_data(self):
        if "p1" in self.data and "p2" in self.data:
            self.update(self.data["p1"], self.data["p2"])

class WindowManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WindowManager, cls).__new__(cls)
            cls._instance.windows = {}
            cls._instance._next_id = 1
            cls._instance.active_tool = "pointer" # Global tool state
        return cls._instance

    def register(self, window):
        """Register a window and return its assigned ID."""
        wid = self._next_id
        self.windows[wid] = window
        self._next_id += 1
        return wid

    def unregister(self, window):
        """Unregister a window instance."""
        # Find ID by value
        for wid, w in list(self.windows.items()):
            if w == window:
                del self.windows[wid]
                return

    def get(self, wid):
        """Get window by ID."""
        return self.windows.get(wid)

    def get_all(self):
        """Return dict of all windows."""
        return self.windows

# Global instance
manager = WindowManager()

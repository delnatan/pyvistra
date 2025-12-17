import sys
import numpy as np
from qtpy.QtWidgets import QApplication
from impy.ui import ImageWindow
from impy.widgets import ChannelPanel


def verify_channel_panel():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    print("Verifying Channel Panel Implementation...")

    # Create synthetic multi-channel data (T=1, Z=5, C=3, Y=100, X=100)
    np.random.seed(42)
    data = np.random.rand(1, 5, 3, 100, 100).astype(np.float32) * 1000

    win = ImageWindow(data, title="Test Multi-Channel")
    win.show()

    # 1. Test channel visibility in renderer
    print("\n1. Testing CompositeImageVisual visibility methods...")
    renderer = win.renderer

    # All channels should be visible by default
    for c in range(3):
        if not renderer.get_channel_visible(c):
            print(f"FAIL: Channel {c} should be visible by default")
            return
    print("PASS: All channels visible by default")

    # Toggle visibility
    renderer.set_channel_visible(1, False)
    if renderer.get_channel_visible(1):
        print("FAIL: Channel 1 should be hidden after set_channel_visible(1, False)")
        return
    print("PASS: set_channel_visible works correctly")

    # Restore visibility
    renderer.set_channel_visible(1, True)

    # 2. Test ChannelPanel creation
    print("\n2. Testing ChannelPanel creation...")
    if win.channel_panel is not None:
        print("FAIL: channel_panel should be None initially")
        return
    print("PASS: channel_panel is None initially")

    win.show_channel_panel()

    if win.channel_panel is None:
        print("FAIL: channel_panel should be created after show_channel_panel()")
        return
    print("PASS: ChannelPanel created successfully")

    if not isinstance(win.channel_panel, ChannelPanel):
        print(f"FAIL: channel_panel is not ChannelPanel, got {type(win.channel_panel)}")
        return
    print("PASS: channel_panel is ChannelPanel instance")

    # 3. Test ChannelPanel has correct number of rows
    print("\n3. Testing ChannelPanel structure...")
    panel = win.channel_panel
    if len(panel.channel_rows) != 3:
        print(f"FAIL: Expected 3 channel rows, got {len(panel.channel_rows)}")
        return
    print("PASS: ChannelPanel has 3 channel rows")

    # 4. Test visibility toggle via panel
    print("\n4. Testing visibility toggle via panel...")
    row = panel.channel_rows[0]

    # Uncheck visibility
    row.chk_visible.setChecked(False)

    # Check renderer state
    if renderer.get_channel_visible(0):
        print("FAIL: Channel 0 should be hidden after unchecking checkbox")
        return
    print("PASS: Visibility toggle updates renderer")

    # Check checkbox state synchronization
    row.chk_visible.setChecked(True)
    if not renderer.get_channel_visible(0):
        print("FAIL: Channel 0 should be visible after checking checkbox")
        return
    print("PASS: Visibility re-toggle works")

    # 5. Test contrast adjustment via panel
    print("\n5. Testing contrast adjustment via panel...")
    row = panel.channel_rows[1]

    # Get initial clim
    initial_clim = renderer.get_clim(1)
    print(f"Initial clim for channel 1: {initial_clim}")

    # Simulate clim change
    new_min, new_max = 100.0, 800.0
    row.histogram.clim_min = new_min
    row.histogram.clim_max = new_max
    row.histogram.climChanged.emit(new_min, new_max)

    # Check renderer state
    updated_clim = renderer.get_clim(1)
    if abs(updated_clim[0] - new_min) > 0.01 or abs(updated_clim[1] - new_max) > 0.01:
        print(f"FAIL: Expected clim ({new_min}, {new_max}), got {updated_clim}")
        return
    print("PASS: Contrast adjustment updates renderer")

    # 6. Test auto-contrast
    print("\n6. Testing auto-contrast all...")
    panel._auto_contrast_all()

    # Just verify it runs without error and changes clim
    new_clim = renderer.get_clim(0)
    print(f"Auto-contrast clim for channel 0: {new_clim}")
    print("PASS: Auto-contrast runs without error")

    # 7. Test menu shortcut exists
    print("\n7. Testing menu integration...")
    # Check that Shift+H is registered
    found_action = False
    for action in win.menuBar().actions():
        menu = action.menu()
        if menu:
            for act in menu.actions():
                if act.text() == "Channels..." and act.shortcut().toString() == "Shift+H":
                    found_action = True
                    break

    if not found_action:
        print("FAIL: Channels... action with Shift+H shortcut not found")
        return
    print("PASS: Channels menu action found with Shift+H shortcut")

    print("\n" + "=" * 50)
    print("All Channel Panel tests passed!")
    print("=" * 50)

    win.close()
    panel.close()


if __name__ == "__main__":
    verify_channel_panel()

"""Test for segfault when quitting without opening an image.

This test verifies that the application can quit cleanly in both scenarios:
1. Quit immediately without opening any image (previously caused segfault)
2. Quit after opening an image (always worked)

Segfaults result in exit code -11 (SIGSEGV) on Linux.
"""

import os
import signal
import subprocess
import sys
import tempfile

import numpy as np


def run_quit_test(
    open_image: bool, timeout: float = 5.0
) -> tuple[int, str, str]:
    """
    Run the app and quit after a short delay.

    Args:
        open_image: If True, create and open a test image before quitting
        timeout: Maximum time to wait for the process

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """

    if open_image:
        # Create a temporary test image
        test_script = """
import sys
import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

# Must create QApplication before importing pyvistra UI components
app = QApplication(sys.argv)

from pyvistra.ui import Toolbar, imshow
from pyvistra.theme import DARK_THEME

app.setStyleSheet(DARK_THEME)

# Create toolbar
toolbar = Toolbar()
toolbar.show()

# Create a small test image and show it
data = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)

def open_and_quit():
    viewer = imshow(data, title="Test Image", dims="zyx")
    # Schedule quit after image is shown
    QTimer.singleShot(500, app.quit)

# Open image after toolbar is shown
QTimer.singleShot(200, open_and_quit)

sys.exit(app.exec_())
"""
    else:
        # Just show toolbar and quit without opening any image
        test_script = """
import sys
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

# Must create QApplication before importing pyvistra UI components
app = QApplication(sys.argv)

from pyvistra.ui import Toolbar
from pyvistra.theme import DARK_THEME

app.setStyleSheet(DARK_THEME)

# Create toolbar
toolbar = Toolbar()
toolbar.show()

# Quit after a short delay (no image opened)
QTimer.singleShot(500, app.quit)

sys.exit(app.exec_())
"""

    # Run in subprocess
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "QT_QPA_PLATFORM": "offscreen",
        },  # Use offscreen for headless testing
    )

    return result.returncode, result.stdout, result.stderr


def exit_code_to_signal_name(code: int) -> str:
    """Convert negative exit code to signal name."""
    if code >= 0:
        return f"exit({code})"

    sig_num = -code
    try:
        sig_name = signal.Signals(sig_num).name
        return f"{sig_name} (exit code {code})"
    except ValueError:
        return f"signal {sig_num} (exit code {code})"


def test_quit_without_image():
    """Test that quitting without opening an image does not segfault."""
    print("Testing: Quit WITHOUT opening an image...")

    try:
        exit_code, stdout, stderr = run_quit_test(open_image=False)
    except subprocess.TimeoutExpired:
        print("  FAIL: Process timed out")
        return False

    if exit_code == -signal.SIGSEGV:
        print(
            f"  FAIL: Segmentation fault detected! ({exit_code_to_signal_name(exit_code)})"
        )
        if stderr:
            print(f"  stderr: {stderr[:500]}")
        return False
    elif exit_code != 0:
        print(
            f"  WARN: Non-zero exit code: {exit_code_to_signal_name(exit_code)}"
        )
        if stderr:
            print(f"  stderr: {stderr[:500]}")
        # Some Qt warnings may cause non-zero exit, but not segfault
        return exit_code != -signal.SIGSEGV
    else:
        print("  PASS: Clean exit")
        return True


def test_quit_with_image():
    """Test that quitting after opening an image does not segfault."""
    print("Testing: Quit AFTER opening an image...")

    try:
        exit_code, stdout, stderr = run_quit_test(open_image=True)
    except subprocess.TimeoutExpired:
        print("  FAIL: Process timed out")
        return False

    if exit_code == -signal.SIGSEGV:
        print(
            f"  FAIL: Segmentation fault detected! ({exit_code_to_signal_name(exit_code)})"
        )
        if stderr:
            print(f"  stderr: {stderr[:500]}")
        return False
    elif exit_code != 0:
        print(
            f"  WARN: Non-zero exit code: {exit_code_to_signal_name(exit_code)}"
        )
        if stderr:
            print(f"  stderr: {stderr[:500]}")
        return exit_code != -signal.SIGSEGV
    else:
        print("  PASS: Clean exit")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("Segfault Regression Test")
    print("=" * 60)
    print()

    results = []

    # Test without image (this was the problematic case)
    results.append(("Quit without image", test_quit_without_image()))
    print()

    # Test with image (this always worked)
    results.append(("Quit with image", test_quit_with_image()))
    print()

    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)

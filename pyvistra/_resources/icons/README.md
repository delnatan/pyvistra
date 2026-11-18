Custom toolbar icons
====================

Drop icon files here to override toolbar icons.

Supported formats:
- `.svg` (preferred)
- `.png`

Naming:
- `pointer.svg` or `pointer.png`
- `rect.svg` or `rect.png`
- `circle.svg` or `circle.png`
- `line.svg` or `line.png`
- `polyline.svg` or `polyline.png`
- `point.svg` or `point.png`
- `brush.svg` or `brush.png`
- `eraser.svg` or `eraser.png`
- `annotations.svg` or `annotations.png`
- `layers.svg` or `layers.png`
- `console.svg` or `console.png`

If a custom file is missing, pyvistra falls back to a built-in Qt icon.

Theme glyphs (not toolbar icons)
--------------------------------

`chevron-up.svg`, `chevron-down.svg` and their `-disabled` variants are the
spin-box and combo-box arrows `pyvistra/theme.py` draws. Their stroke colors
are `colors.TEXT_SECONDARY` and `colors.TEXT_DISABLED`; change them with
those tokens.

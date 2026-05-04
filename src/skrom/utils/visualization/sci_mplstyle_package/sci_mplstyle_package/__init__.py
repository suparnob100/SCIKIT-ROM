"""Bundled Matplotlib style loader.

TL;DR
-----
This module applies the packaged publication Matplotlib style when imported.

Notes
-----
It resolves the style file next to the package and passes it to Matplotlib.
"""

# __init__.py
import matplotlib.pyplot as plt
import os

style_path = os.path.join(os.path.dirname(__file__), 'style_files', 'publication.mplstyle')
plt.style.use(style_path)

import time
import chaospy

project = 'chaospy'
author = 'Jonathan Feinberg'
copyright = '%d, Jonathan Feinberg' % time.gmtime().tm_year
version = ".".join(chaospy.__version__.split(".")[:2])
release = chaospy.__version__
master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'unsrt'

templates_path = ['_templates']
exclude_patterns = ['.build']

rst_prolog = """
"""
language = "en"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "tango"

# Execute content of Jupyter notebooks:
# "always", "never", "auto" (on empty cell only)
nbsphinx_execute = "never"

# Create stubs automatically for all auto-summaries:
autosummary_generate = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
    "external_links":
        [{"name": "Github", "url": "https://github.com/jonathf/chaospy"}],
    "footer_items": ["sphinx-version.html"],
    "navbar_align": "left",
    "navbar_end": ["search-field.html"],
    "navigation_depth": 2,
    "show_prev_next": False,
}
html_short_title = "chaospy"
html_context = {
    "doc_path": "docs",
}
html_logo = "_static/chaospy_logo2.svg"
html_static_path = ['_static']
html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
}

htmlhelp_basename = 'chaospy'
html_show_sourcelink = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'numpoly': ('https://numpoly.readthedocs.io/en/master/', None),
    'openturns': ('https://openturns.github.io/openturns/master/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
}

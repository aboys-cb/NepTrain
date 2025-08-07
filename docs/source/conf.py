# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'NepTrain'
copyright = '2024, NepTrain Team'
author = ",".join(['ChengBing Chen','YuTong Li'])
release = '1.5.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
 

html_show_sourcelink = False
extensions = [
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.napoleon',
    # 'sphinx.ext.mathjax',

"sphinx_design",
    "myst_parser"
]
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
# locale_dirs = ['docs/locales']  # directory for language files
language = 'en'


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_context = {
    "author_name": author,
}
html_css_files = [
    'css/custom.css',  # specify your custom CSS file here
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

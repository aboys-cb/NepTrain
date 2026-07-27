from importlib.metadata import version as distribution_version
import os


project = "NepTrain"
copyright = "2024-2026, NepTrain Team"
author = ", ".join(["ChengBing Chen", "YuTong Li"])
release = distribution_version("NepTrain")


html_show_sourcelink = False
extensions = [
    "sphinx_design",
    "myst_parser",
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

templates_path = ["_templates"]
locale_dirs = ["locale/"]
gettext_compact = False
gettext_uuid = True
gettext_additional_targets = {"image", "literal-block"}
_rtd_language = os.environ.get("READTHEDOCS_LANGUAGE", "").lower()
language = {
    "zh-cn": "zh_CN",
    "zh_cn": "zh_CN",
}.get(_rtd_language, _rtd_language or "zh_CN")


html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_context = {
    "author_name": author,
}
html_css_files = [
    "css/custom.css",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

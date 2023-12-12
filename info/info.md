## Environment Setup
1. Install d2lbook.
```bash
git clone https://github.com/openmlsys/d2l-book.git
cd d2l-book
pip install -e .
```
The d2lbook needs python `pandoc` package. Use `conda install pandoc` to install. You can use `Homebrew` in MacOS.
If you have SVG pictures while constructing the PDF, please install `librsvg` by `apt`. In MacOS, please use `Homebrew`.
LATEX is necessary while constructing the PDF. Please install it on your computer.

## Compiling HTML
```bash
 git clone https://github.com/openmlsys/openmlsys-en.git
 cd openmlsys-en
 sh build_html.sh
```
The generated HTML files will be at `_build/html`.
At this time, please copy the contents of the entire folder of the compiled HTML to the 'docs' directory of 'openmlsys.github.io'.
It is important to note that the .nojekyll file in the docs directory should not be deleted, otherwise the webpage will not render.
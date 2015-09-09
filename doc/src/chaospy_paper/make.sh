#!/bin/bash
set -x
name=chaospy

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

system doconce format pdflatex $name --device=screen "--latex_code_style=default:vrb-blue1@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" --latex_copyright=titlepages
system pdflatex $name
system bibtex $name
# makeindex $name
pdflatex $name
pdflatex $name
cp ${name}.pdf ${name}-4screen.pdf

system doconce format pdflatex $name --device=paper "--latex_code_style=default:vrb-blue1@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" --latex_copyright=titlepages
system pdflatex $name
bibtex $name
# makeindex $name
pdflatex $name
pdflatex $name
cp ${name}.pdf ${name}-4print.pdf

style=solarized
system doconce format html $name --html_style=solarized3 --html_output=${name}-${style} --pygments_html_style=perldoc
system doconce split_html ${name}-${style}.html

style=bootstrap
system doconce format html $name --html_style=bootswatch_readable --html_output=${name}-${style} --html_code_style=inherit
system doconce split_html ${name}-${style}.html

system doconce format sphinx $name
system doconce split_rst $name
system doconce sphinx_dir theme=cbc $name
system python automake_sphinx.py

# Publish
dest=../../pub
cp *.pdf $dest
cp ${name}-*.html ._${name}*.html $dest
cp fig/* $dest/fig
rm -rf $dest/sphinx
cp -r sphinx-rootdir/_build/html $dest/sphinx
doconce format html index --html_style=bootstrap --html_bootstrap_navbar=off
cp index.html $dest

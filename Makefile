notes/tut%.pdf: src/tut%.md
	cd src && \
	pandoc tut$*.md -s -o ../notes/tut$*.pdf -V colorlinks=true -V linkcolor=red -V urlcolor=blue -V toccolor=gray

slides/slides_tut%.pdf: src/tut%.md
	cd src && \
		pandoc -s --dpi=300 --slide-level 2 --toc --listings --shift-heading-level=0 -V classoption:aspectratio=169 -V theme:Berlin -t beamer tut$*.md -o ../slides/slides_tut$*.pdf

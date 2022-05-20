notes/tut%.pdf: src/tut%.md
	cd src && \
	pandoc tut$*.md -s -o ../notes/tut$*.pdf -V colorlinks=true -V linkcolor=red -V urlcolor=blue -V toccolor=gray

slides/slides_tut%.pdf: src/tut%.md
	cd src && \
	pandoc -t beamer tut$*.md -s -o ../slides/slides_tut$*.pdf

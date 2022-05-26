TEXS = $(wildcard src/*.md)
SLIDES = $(patsubst src/%.md, slides/slides_%.pdf, $(TEXS))
NOTES = $(patsubst src/%.md, notes/%.pdf, $(TEXS))

all: $(SLIDES)
	echo $(TEXS)
notes: $(NOTES)

slides/slides_%.pdf: src/%.md
	cd src && \
	pandoc -s --dpi=300 --slide-level 2 --toc --listings --shift-heading-level=0 -V classoption:aspectratio=169 -V theme:Berlin -t beamer $*.md -o ../slides/slides_$*.pdf

notes/%.pdf: src/%.md
	cd src && \
	pandoc tut$*.md -s -o ../notes/tut$*.pdf -V colorlinks=true -V linkcolor=red -V urlcolor=blue -V toccolor=gray

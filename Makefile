main.pdf : *.tex
	pdflatex main

full: *.tex
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main


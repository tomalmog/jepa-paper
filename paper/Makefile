PAPER = main

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib neurips_2024.sty
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc $(PAPER).bbl $(PAPER).blg

distclean: clean
	rm -f $(PAPER).pdf

.PHONY: all clean distclean

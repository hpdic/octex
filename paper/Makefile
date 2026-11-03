TARGET = octex
PDF = $(TARGET).pdf
TEX = $(TARGET).tex
BIB = refs.bib
BST = acm.bst

.PHONY: all clean pdf check arxiv

all: pdf

pdf: $(PDF)

$(PDF): $(TEX) $(BIB) $(BST) usenix.sty $(wildcard figures/*.pdf)
	latexmk -g -pdf -interaction=nonstopmode -synctex=1 $(TARGET).tex

clean:
	latexmk -C $(TARGET).tex
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk *.synctex.gz

check: pdf
	@echo "Checking PDF page count and layout..."
	@pdfinfo $(PDF) | grep -E "Pages:|Page size:"

# arXiv source: the tex, the bbl that arXiv needs instead of running bibtex, the style files and the figures
arxiv: pdf
	rm -rf arxiv && mkdir -p arxiv/figures
	cp $(TEX) $(TARGET).bbl usenix.sty usenix-2020-09.sty $(BST) arxiv/
	sed -i '/^[[:space:]]*%/d' arxiv/$(TEX)  # arXiv publishes the source: drop commented-out lines
	cp figures/*.pdf arxiv/figures/
	tar -czf $(TARGET)-arxiv.tar.gz -C arxiv .
	@echo "wrote $(TARGET)-arxiv.tar.gz"

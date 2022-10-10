paper: paper.md template.tex lit_review/sources.bib
	pandoc -o paper.pdf paper.md --filter pandoc-citeproc  --template template.tex
paper_tex: paper.md template.tex lit_review/sources.bib
	pandoc -o paper_debug.tex paper.md --filter pandoc-citeproc --template template.tex
clean:
	rm *.out; rm *.gz; rm *.fls; rm *.log; rm *.fdb_latexmk; rm *.aux; rm paper_debug.tex; rm paper_debug.pdf; rm *.toc
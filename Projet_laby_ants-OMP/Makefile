# Makefile principal
# 

all: gui project
all_doc: gui_doc project_doc

clean_all:
	(cd libgui; make clean)
	(cd project; make clean)
	@rm -rf include lib

clobber: clean_all
	@rm -fr include lib test
	@(cd libgui; rm -fr doc)
	@(cd project; rm -fr doc)

gui:
	@mkdir -p lib
	@mkdir -p include
	(cd libgui; make)
	(cd project; make)

gui_doc:
	@mkdir -p lib
	@mkdir -p include
	(cd libgui; make doc)
	(cd project; make doc)

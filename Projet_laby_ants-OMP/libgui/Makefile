include ../Make.inc

SRC_DIR = src
INCL_DIR = $(PROJECT_ROOT)/include/gui
OBJ_DIR = build
LIB_DIR = $(PROJECT_ROOT)/lib
TST_DIR = $(PROJECT_ROOT)/test

HEADERS = \
	$(INCL_DIR)/colors.hpp $(INCL_DIR)/context.hpp $(INCL_DIR)/event_manager.hpp \
	$(INCL_DIR)/point.hpp $(INCL_DIR)/quad.hpp $(INCL_DIR)/segment.hpp $(INCL_DIR)/triangle.hpp \
	$(INCL_DIR)/window.hpp

SRCS = \
	$(SRC_DIR)/context.cpp $(SRC_DIR)/event_manager.cpp $(SRC_DIR)/window.cpp

OBJS = \
	$(OBJ_DIR)/context.o $(OBJ_DIR)/event_manager.o $(OBJ_DIR)/window.o

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	@echo "\033[33m Compiling " $@ "\033[0m"
	@$(CXX) $(CXXOPT) -I$(PROJECT_ROOT)/libgui/include -I$(INCL_DIR) -I$(PROJECT_ROOT)/include -I$(OPENGLINC) -c $< -o $@

all: thirdparty $(PROJECT_ROOT)/include $(LIB_DIR)/libgui.so test

$(PROJECT_ROOT)/include:
	@mkdir -p $(PROJECT_ROOT)/include
	@echo "\033[31m Copying includes\033[0m"
	@rsync -rpuE --delete  include/gui $(PROJECT_ROOT)/include/

$(LIB_DIR)/libgui.so: $(OBJS)
	@echo "\033[35m Linking " $@ "\033[0m"
	$(CXX) -shared $(CXXOPT) -L$(OPENGLLIBROOT)  -o $@ $(OBJS) -L$(LIB_DIR)  -Wl,-rpath=$(LIB_DIR) --enable-new-dtags -lglut -lGLU -lGL 

thirdparty: $(LIB_DIR)/libglut.so
$(LIB_DIR)/libglut.so: thirdparty/freeglut-3.0.0.tar.gz
	(cd thirdparty; sh ./install.sh)

doc:
	doxygen Doxyfile.in

test: $(TST_DIR)/test_rendering.exe

$(TST_DIR)/test_rendering.exe: $(LIB_DIR)/libgui.so test/test_rendering.cpp
	@mkdir -p $(TST_DIR)
	@echo "\033[34m Compiling " $@ "\033[0m"
	@$(CXX) $(CXXOPT) -I$(OPENGLINC) -L$(OPENGLLIBROOT) -L$(LIB_DIR)  -I$(PROJECT_ROOT)/include -o $@ $^ -Wl,-rpath=$(LIB_DIR)  -lgui -lglut -lGLU -lGL

clean:
	@rm -rf $(OBJ_DIR)/
	@rm -f $(SRC_DIR)/*~
	@rm -rf $(INCL_DIR)/
	@rm -rf $(TST_DIR)/
	@rm -rf thirdparty/build thirdparty/freeglut-3.0.0

# Les dépendances :
$(OBJ_DIR)/context.o: $(INCL_DIR)/context.hpp $(SRC_DIR)/context.cpp 
$(OBJ_DIR)/event_manager.o: $(INCL_DIR)/event_manager.hpp $(SRC_DIR)/event_manager.cpp
$(OBJ_DIR)/window.o: $(INCL_DIR)/window.hpp $(SRC_DIR)/window.cpp

.PHONY: thirdparty doc $(PROJECT_ROOT)/include

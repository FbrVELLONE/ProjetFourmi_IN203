# include <vector>
# include "gui/context.hpp"
# include "gui/colors.hpp"
# include "gui/point.hpp"
# include "gui/segment.hpp"
# include "gui/triangle.hpp"
# include "gui/quad.hpp"
# include "gui/event_manager.hpp"

int main(int nargs, char* argv[])
{
	gui::context graphic_context(nargs, argv);
	gui::window& win =  graphic_context.new_window(640,480);

    std::vector<gui::segment> lines;
    for ( int i = 0; i < 640; i += 2 ){
    	lines.emplace_back( gui::point(i,479, 0, gui::color::rgba({(640-i)/640.,i/640.,0.})), 
    		                gui::point(639-i,0, 0, gui::color::rgba({0., (640-i)/640.,i/640.})) );
    }
    std::vector<gui::triangle> trigs = { gui::triangle{ gui::point(0,0,   0, gui::color::rgba({1.,0.,0.})), 
                                                        gui::point(0,479, 0, gui::color::rgba({0., 1., 0.})),
                                                        gui::point(319,239,0,gui::color::rgba({0.5,0.5,0.}))} 
                                       };
    std::vector<gui::quad> quads = { gui::quad{ gui::point(639, 159, 0, gui::color::rgba({1.,0.,0.})), 
                                                gui::point(639, 319, 0, gui::color::rgba({0., 1., 0.})),
                                                gui::point(609, 319, 0, gui::color::rgba({0.5,0.5,0.})),
                                                gui::point(609, 159, 0, gui::color::rgba({0.5,0.5,0.5}))}
                                   };

    bool quitting = false;
    gui::event_manager manager;
    manager.on_key_event(int('q'), [&quitting] (int code) { exit(0); });
    manager.on_display([&win,&lines, &trigs, &quads] () { win.clear(); gui::draw(lines); gui::draw(trigs); gui::draw(quads); win.blit(); });
    manager.loop();
	return EXIT_SUCCESS;
}
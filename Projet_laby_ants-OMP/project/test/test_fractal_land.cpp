# include <iostream>
# include "gui/context.hpp"
# include "gui/colors.hpp"
# include "gui/point.hpp"
# include "gui/segment.hpp"
# include "gui/triangle.hpp"
# include "gui/quad.hpp"
# include "gui/event_manager.hpp"
# include "fractal_land.hpp"

int main(int nargs, char** argv)
{
    fractal_land land(8,2,1.,1024);
    gui::context graphic_context(nargs, argv);
    gui::window& win =  graphic_context.new_window(512,512);
    std::vector<gui::point> points;
    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )  {
            double c= (land(i,j)-min_val)/delta;
            points.push_back(gui::point(i,j, 0, gui::color::rgba({c,c,c})));
        }
    std::cout << std::endl;
    bool quitting = false;
    gui::event_manager manager;
    manager.on_key_event(int('q'), [&quitting] (int code) { exit(0); });
    manager.on_display([&win,&points] () { win.clear(); gui::draw(points); win.blit(); });
    manager.loop();
    return EXIT_SUCCESS;
}
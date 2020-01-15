#ifndef _DISPLAY_HPP_
# define _DISPLAY_HPP_
#include <utility>
# include <vector>
# include "labyrinthe.hpp"
# include "pheronome.hpp"
# include "ant.hpp"
# include "gui/window.hpp"
# include "gui/segment.hpp"

struct display_t
{
    display_t( const labyrinthe& laby, const pheronome& phen, 
               const position_t& pos_nest, const position_t& pos_food,
               const std::vector<ant>& ants, gui::window& win );
    display_t(const display_t& ) = delete;
    ~display_t()=default;

    void display( const std::size_t& compteur );
private:
    struct graph_context;
    const labyrinthe& m_ref_laby;
    const pheronome& m_ref_phen;
    const position_t& m_pos_nest;
    const position_t& m_pos_food;
    const std::vector<ant>& m_ref_ants;
    std::vector<std::size_t> m_curve;    
    gui::window& m_ref_win;
    // La carte territoriale à afficher :
    std::vector<gui::segment> laby_points;
};


#endif
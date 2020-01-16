#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include "labyrinthe.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
# include "gui/context.hpp"
# include "gui/colors.hpp"
# include "gui/point.hpp"
# include "gui/segment.hpp"
# include "gui/triangle.hpp"
# include "gui/quad.hpp"
# include "gui/event_manager.hpp"
# include "display.hpp"
#include <fstream>
void advance_time( const labyrinthe& land, pheronome& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur )
{
    for ( size_t i = 0; i < ants.size(); ++i )
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    phen.do_evaporation();
    phen.update();
}

int main(int nargs, char* argv[])
{
    bool victoire = true;
    std::chrono::time_point<std::chrono::system_clock> victStart, victSEnd;
    victStart = std::chrono::system_clock::now();

    const dimension_t dims{32, 64};// Dimension du labyrinthe
    const std::size_t life = int(dims.first*dims.second);
    const int nb_ants = 2*dims.first*dims.second; // Nombre de fourmis
    const double eps = 0.75;  // Coefficient d'exploration
    const double alpha=0.97; // Coefficient de chaos
    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta=0.999; // Coefficient d'évaporation
                             // 
    labyrinthe laby(dims);
    // Location du nid
    position_t pos_nest{dims.first/2,dims.second/2};
    // Location de la nourriture
    position_t pos_food{dims.first-1,dims.second-1};
                          
    
    // Définition du coefficient d'exploration de toutes les fourmis.
    ant::set_exploration_coef(eps);
    // On va créer toutes les fourmis dans le nid :
    std::vector<ant> ants;
    ants.reserve(nb_ants);
    for ( size_t i = 0; i < nb_ants; ++i )
        ants.emplace_back(pos_nest, life);
    // On crée toutes les fourmis dans la fourmilière.
    pheronome phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);

    gui::context graphic_context(nargs, argv);
    gui::window& win =  graphic_context.new_window(h_scal*laby.dimensions().second,h_scal*laby.dimensions().first+266);
    display_t displayer( laby, phen, pos_nest, pos_food, ants, win );
    size_t food_quantity = 0;

    gui::event_manager manager;
    manager.on_key_event(int('q'), [] (int code) { exit(0); });
    manager.on_display([&] { displayer.display(food_quantity); win.blit(); });
    manager.on_idle([&] () { 
        advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);
        displayer.display(food_quantity); 
        win.blit(); 

        if (food_quantity >= 5000 && victoire){
            victSEnd = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = victSEnd - victStart;
            std::ofstream outfile ("saida.txt", std::ios::app	);

            outfile << "Victoire seq: " << duration.count() << std::endl;

            outfile.close();
            victoire = false;;
        }

    });
    manager.loop();

    return 0;
}
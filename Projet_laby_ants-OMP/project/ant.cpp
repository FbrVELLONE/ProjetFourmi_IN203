#include "ant.hpp"
#include <iostream>
#include <random>

double ant::m_eps = 0.;

void ant::advance( pheronome& phen, const labyrinthe& laby, const position_t& pos_food, const position_t& pos_nest,
                   std::size_t& cpteur_food ) {
    std::random_device                       rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937                             gen( rd( ) );  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution< double > ant_choice( 0., 1. );
    std::uniform_int_distribution< int >     dir_choice( 1, 4 );
    // Si la fourmi est chargée, elle revient simplement à son nid :
    if ( is_loaded( ) ) {
        position_t p = return_to_nest( );
        if ( p == pos_nest ) {
            unset_loaded( );
            m_path_to_nest.resize( 0 );
            m_path_to_nest.push_back( pos_nest );
            cpteur_food += 1;
        }
    } else {
        // Si la fourmi n'est pas chargée, elle suit les phéromones ou elle explore
        // Mais elle perd aussi de l'énergie :
        cur_life -= 1;
        if ( cur_life == 0)
        {
            cur_life = max_life;
            m_path_to_nest.resize(0);
            m_path_to_nest.emplace_back(pos_nest);
            return;
        }
       double     choix       = ant_choice( gen );  // Le choix entre exploration ou suivre des phéronomes :
        position_t old_pos_ant = get_position( );
        position_t new_pos_ant = old_pos_ant;
        // Calcul des phéronomes adjacents en tenant compte des ouvertures du labyrinthe
        // Les phéronomes ne traversent pas les murs !
        double west_phen =
            ( laby( old_pos_ant ) & labyrinthe::WEST ? phen( new_pos_ant.first, new_pos_ant.second - 1 ) : 0. );
        double east_phen =
            ( laby( old_pos_ant ) & labyrinthe::EAST ? phen( new_pos_ant.first, new_pos_ant.second + 1 ) : 0. );
        double south_phen =
            ( laby( old_pos_ant ) & labyrinthe::SOUTH ? phen( new_pos_ant.first - 1, new_pos_ant.second ) : 0. );
        double north_phen =
            ( laby( old_pos_ant ) & labyrinthe::NORTH ? phen( new_pos_ant.first + 1, new_pos_ant.second ) : 0. );
        double max_phen = std::max( {west_phen, east_phen, south_phen, north_phen} );
        double nb_exist =
            ( (laby( old_pos_ant ) & labyrinthe::WEST) > 0 ) + ( (laby( old_pos_ant ) & labyrinthe::EAST) > 0 ) +
            ( (laby( old_pos_ant ) & labyrinthe::SOUTH) > 0 ) + ( (laby( old_pos_ant ) & labyrinthe::NORTH) > 0 );
        if ( ( choix > m_eps ) || ( max_phen == 0. ) ) {
            bool move_is_valid = false;
            while ( move_is_valid == false ) {
                new_pos_ant = old_pos_ant;
                int d       = dir_choice( gen );
                if ( ( d == 1 ) && ( laby( old_pos_ant ) & labyrinthe::WEST ) ) {
                    new_pos_ant.second -= 1;
                    if ((nb_exist == 1) || (m_path_to_nest[m_path_to_nest.size()-2] != new_pos_ant))
                        move_is_valid = true;
                }
                if ( ( d == 2 ) && ( laby( old_pos_ant ) & labyrinthe::SOUTH ) ) {
                    new_pos_ant.first -= 1;
                    if ((nb_exist == 1) || (m_path_to_nest[m_path_to_nest.size()-2] != new_pos_ant))
                    move_is_valid = true;
                }
                if ( ( d == 3 ) && ( laby( old_pos_ant ) & labyrinthe::EAST ) ) {
                    new_pos_ant.second += 1;
                    if ((nb_exist == 1) || (m_path_to_nest[m_path_to_nest.size()-2] != new_pos_ant))
                    move_is_valid = true;
                }
                if ( ( d == 4 ) && ( laby( old_pos_ant ) & labyrinthe::NORTH ) ) {
                    new_pos_ant.first += 1;
                    if ((nb_exist == 1) || (m_path_to_nest[m_path_to_nest.size()-2] != new_pos_ant))
                    move_is_valid = true;
                }
            }
        } else {
            // On choisit la case où le phéromone est le plus fort.
            if ( west_phen == max_phen )
                new_pos_ant.second -= 1;
            else if ( east_phen == max_phen )
                new_pos_ant.second += 1;
            else if ( south_phen == max_phen )
                new_pos_ant.first -= 1;
            else  // if (north_phen == max_phen)
                new_pos_ant.first += 1;
        }
        phen.mark_pheronome( new_pos_ant, laby );
        m_path_to_nest.emplace_back( new_pos_ant );
        if ( get_position( ) == pos_food ) { set_loaded( ); }
        if (get_position() == pos_nest) { cur_life = max_life; }
    }
}
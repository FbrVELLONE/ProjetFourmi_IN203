#include "labyrinthe.hpp"
#include <iostream>
#include <random>

labyrinthe::labyrinthe( const dimension_t& dims ) : m_dims( dims ), m_laby_data( dims.first * dims.second, 0 ) {
    std::vector< char >    is_visited( dims.first * dims.second, 0 );
    std::vector< position_t > historic;
    historic.reserve( dims.first * dims.second );
    std::random_device                   rd;            // Will be used to obtain a seed for the random number engine
    std::mt19937                         gen( rd( ) );  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution< int > dir_choice( 1, 4 );


    // On choisit comme cellule initiale une cellule centrale
    position_t cur_ind{dims.first / 2, dims.second / 2};
    is_visited[comp_flat_index( cur_ind )] = 1;
    historic.push_back( cur_ind );
    while ( not historic.empty( ) ) {
        cur_ind = historic.back( );
        // On regarde en premier lieu si il existe au moins une cellule non visitée voisine de la cellule
        // courante :
        std::size_t cur_flat_index   = comp_flat_index( cur_ind );
        std::size_t west_flat_index  = comp_flat_index( {cur_ind.first, cur_ind.second-1} );
        std::size_t east_flat_index  = comp_flat_index( {cur_ind.first, cur_ind.second+1} );
        std::size_t south_flat_index = comp_flat_index( {cur_ind.first-1, cur_ind.second} );
        std::size_t north_flat_index = comp_flat_index( {cur_ind.first+1, cur_ind.second} );
        int         nb_vis_cells     = ( cur_ind.second > 0 ? is_visited[west_flat_index] : 1 ) +
                           ( cur_ind.second < m_dims.second - 1 ? is_visited[east_flat_index] : 1 ) +
                           ( cur_ind.first > 0 ? is_visited[south_flat_index] : 1 ) +
                           ( cur_ind.first < m_dims.first - 1 ? is_visited[north_flat_index] : 1 );
        if ( nb_vis_cells < 4 )  // Dans ce cas, au moins une cellule est non visitée
        {
            bool    valid_choice = false;
            int dir;
            // Tant qu'on n'a pas choisi une cellule valide ( car en dehors de la grille ou déjà visitée )
            while ( not valid_choice ) {
                // On choisit au hasard une cellule voisine non visitée :
                dir = dir_choice( gen );
                switch ( dir ) {
                case 1:  // On a choisi la cellule Ouest
                {
                    if ( ( cur_ind.second > 0 ) and ( is_visited[west_flat_index] == 0 ) ) {
                        m_laby_data[cur_flat_index] |= WEST;
                        m_laby_data[west_flat_index] |= EAST;
                        historic.push_back({cur_ind.first,cur_ind.second-1});
                        valid_choice  = true;
                    }
                    break;
                }
                case 2:  // On a choisi la cellule Est
                {
                    if ( ( cur_ind.second < m_dims.second - 1 ) and ( is_visited[east_flat_index] == 0 ) ) {
                        m_laby_data[cur_flat_index] |= EAST;
                        m_laby_data[east_flat_index] |= WEST;
                        historic.push_back({cur_ind.first,cur_ind.second+1});
                        valid_choice  = true;
                    }
                    break;
                }
                case 3:  // On a choisi la cellule Sud
                {
                    if ( ( cur_ind.first > 0 ) and ( is_visited[south_flat_index] == 0 ) ) {
                        m_laby_data[cur_flat_index] |= SOUTH;
                        m_laby_data[south_flat_index] |= NORTH;
                        historic.push_back({cur_ind.first-1,cur_ind.second});
                        valid_choice  = true;
                    }
                    break;
                }
                case 4:  // On a choisi la cellule Nord
                {
                    if ( ( cur_ind.first < m_dims.first - 1 ) and ( is_visited[north_flat_index] == 0 ) ) {
                        m_laby_data[cur_flat_index] |= NORTH;
                        m_laby_data[north_flat_index] |= SOUTH;
                        historic.push_back({cur_ind.first+1,cur_ind.second});
                        valid_choice  = true;
                    }
                    break;
                }
                }  // switch dir
            }// while (not valid_choice)
            is_visited[comp_flat_index(historic.back())] = 1;
        } // Si pas de cellules libres voisines, on retourne sur une cellule précédemment visitée :
        else historic.pop_back();
    }// Tant que l'historique n'est pas vide, on peut trouver une cellule libre voisine d'une cellule
     // précédemment visitée...
}
#ifndef _PHERONOME_HPP_
#define _PHERONOME_HPP_
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include "basic_types.hpp"
#include "labyrinthe.hpp"

/**
 * @brief Carte des phéronomes
 * @details Gère une carte des phéronomes avec leurs mis à jour ( dont l'évaporation )
 *
 */
class pheronome {
public:
    using size_t      = unsigned long;
    using pheronome_t = double;

    /**
     * @brief Construit une carte initiale des phéronomes
     * @details La carte des phéronomes est initialisées à zéro ( neutre )
     *
     * @param dim Nombre de cellule dans chaque direction
     * @param alpha Paramètre de bruit
     * @param beta Paramêtre d'évaporation
     */
    pheronome( dimension_t dim, const position_t& pos_food, const position_t& pos_nest, double alpha = 0.7,
               double beta = 0.9999 )
        : m_dim( dim ),
          m_stride( dim.second ),
          m_alpha( alpha ),
          m_beta( beta ),
          m_map_of_pheronome( dim.first * dim.second, 0. ),
          m_buffer_pheronome( ),
          m_pos_food( pos_food ),
          m_pos_nest( pos_nest ) {
        m_map_of_pheronome[index( pos_food )] = 1.;
        m_buffer_pheronome                    = m_map_of_pheronome;
    }
    pheronome( const pheronome& ) = delete;
    pheronome( pheronome&& )      = delete;
    ~pheronome( )                 = default;

    pheronome_t& operator( )( size_t i, size_t j ) {
        assert( i < m_dim.first );
        assert( i >= 0 );
        assert( j < m_dim.second );
        assert( j >= 0 );
        return m_map_of_pheronome[i * m_stride + j];
    }

    const pheronome_t& operator( )( size_t i, size_t j ) const {
        assert( i < m_dim.first );
        assert( i >= 0 );
        assert( j < m_dim.second );
        assert( j >= 0 );
        return m_map_of_pheronome[i * m_stride + j];
    }

    pheronome_t& operator[]( const position_t& pos ) {
        return m_map_of_pheronome[index( pos )];
    }

    const pheronome_t& operator[]( const position_t& pos ) const {
        return m_map_of_pheronome[index( pos )];
    }

    void do_evaporation( ) {
        for ( decltype(m_dim.first) i = 0; i < m_dim.first; ++i )
            for ( decltype(m_dim.second) j = 0; j < m_dim.second; ++j ) { m_buffer_pheronome[i * m_stride + j] *= m_beta; }
    }

    void mark_pheronome( const position_t& pos, const labyrinthe& laby ) {
        std::size_t i = pos.first;
        std::size_t j = pos.second;
        assert( i >= 0 );
        assert( j >= 0 );
        assert( i < m_dim.first );
        assert( j < m_dim.second );
        pheronome& phen = *this;

        const pheronome_t& left_cell   = (laby({i,j}) & labyrinthe::WEST) ? phen( i, j-1 ) : 0.;
        const pheronome_t& right_cell  = (laby({i,j}) & labyrinthe::EAST) ? phen( i, j+1 ) : 0;
        const pheronome_t& upper_cell  = (laby({i,j}) & labyrinthe::SOUTH) ? phen( i-1, j ) : 0.;
        const pheronome_t& bottom_cell = (laby({i,j}) & labyrinthe::NORTH) ? phen( i+1, j ) : 0;
        double             v1_left     = std::max( left_cell, 0. );
        double             v1_right    = std::max( right_cell, 0. );
        double             v1_upper    = std::max( upper_cell, 0. );
        double             v1_bottom   = std::max( bottom_cell, 0. );
        m_buffer_pheronome[i*m_stride + j] =
            m_alpha * std::max( {v1_left, v1_right, v1_upper, v1_bottom} ) +
            ( 1 - m_alpha ) * 0.25 * ( v1_left + v1_right + v1_upper + v1_bottom );
    }

    void update( ) {
        m_map_of_pheronome.swap( m_buffer_pheronome );
	std::copy(m_map_of_pheronome.begin(), m_map_of_pheronome.end(),
		  m_buffer_pheronome.begin());
        m_map_of_pheronome[m_pos_food.first * m_stride + m_pos_food.second] = 1;
    }

private:
    size_t index( const position_t& pos ) const {
        return pos.first * m_stride + pos.second;
    }
    dimension_t                m_dim;
    unsigned long              m_stride;
    double                     m_alpha, m_beta;
    std::vector< pheronome_t > m_map_of_pheronome, m_buffer_pheronome;
    position_t                 m_pos_food, m_pos_nest;
};

#endif

#include "display.hpp"
#include <iostream>
#include "gui/colors.hpp"
#include "gui/point.hpp"
#include "gui/quad.hpp"
#include "gui/quad.hpp"
#include "gui/segment.hpp"
#include "gui/triangle.hpp"
#include "basic_types.hpp"

const double h = h_scal;

display_t::display_t( const labyrinthe& laby, const pheronome& phen, const position_t& pos_nest,
                      const position_t& pos_food, const std::vector< ant >& ants, gui::window& win )
    : m_ref_laby( laby ),
      m_ref_phen( phen ),
      m_pos_nest( pos_nest ),
      m_pos_food( pos_food ),
      m_ref_ants( ants ),
      m_curve( ),
      m_ref_win( win ),
      laby_points( ) {
    m_curve.reserve( 50000 );
    laby_points.reserve( 4 * laby.dimensions( ).first * laby.dimensions( ).second );
    for ( std::size_t i = 0; i < m_ref_laby.dimensions( ).first; ++i )
        for ( std::size_t j = 0; j < m_ref_laby.dimensions( ).second; ++j ) {
            unsigned char c = m_ref_laby( {i, j} );
            // Affichage des murs :
            if ( not( c & labyrinthe::WEST ) ) {
                laby_points.emplace_back( gui::point{h * j, h * i, 0, gui::color::rgba( {255, 255, 255} )},
                                          gui::point{h * j, h * i + h - 1, 0, gui::color::rgba( {255, 255, 255} )} );
            }
            if ( not( c & labyrinthe::EAST ) ) {
                laby_points.emplace_back(
                    gui::point{h * j + h - 1, h * i, 0, gui::color::rgba( {255, 255, 255} )},
                    gui::point{h * j + h - 1, h * i + h - 1, 0, gui::color::rgba( {255, 255, 255} )} );
            }
            if ( not( c & labyrinthe::SOUTH ) ) {
                laby_points.emplace_back( gui::point{h * j, h * i, 0, gui::color::rgba( {255, 255, 255} )},
                                          gui::point{h * j + h - 1, h * i, 0, gui::color::rgba( {255, 255, 255} )} );
            }
            if ( not( c & labyrinthe::NORTH ) ) {
                laby_points.emplace_back(
                    gui::point{h * j, h * i + h - 1, 0, gui::color::rgba( {255, 255, 255} )},
                    gui::point{h * j + h - 1, h * i + h - 1, 0, gui::color::rgba( {255, 255, 255} )} );
            }
            if ( ( i == pos_food.first ) and ( j == pos_food.second ) ) {
                laby_points.emplace_back(
                    gui::point( h * j + 1, h * i + 1, 0, gui::color::rgba( {0, 255, 0} ) ),
                    gui::point( h * j + h - 2, h * i + h - 2, 0, gui::color::rgba( {0, 255, 0} ) ) );
            }
        }
}

void display_t::display( const std::size_t& compteur ) {
    m_ref_win.clear( );
    // Affichage des paysages
    gui::draw( laby_points );
    // Affichage des phéronomes dans le cadran en haut à droite :
    std::vector< gui::quad > m_p_phen;
    m_p_phen.reserve( m_ref_laby.dimensions( ).first * m_ref_laby.dimensions( ).second+1 );
    for ( std::size_t i = 0; i < m_ref_laby.dimensions( ).first; ++i )
        for ( std::size_t j = 0; j < m_ref_laby.dimensions( ).second; ++j ) {
            m_p_phen.emplace_back(
                gui::point( h * j + 1, h * i + 1, 0.0,
                            gui::color::rgba( {std::min( 1., (double)m_ref_phen( i, j ) ), 0., 0.,0.75} ) ),
                gui::point( h * j + 7, h * i + 1, 0.0,
                            gui::color::rgba( {std::min( 1., (double)m_ref_phen( i, j ) ), 0., 0.,0.75} ) ),
                gui::point( h * j + 7, h * i + 7, 0.0,
                            gui::color::rgba( {std::min( 1., (double)m_ref_phen( i, j ) ), 0., 0.,0.75} ) ),
                gui::point( h * j + 1, h * i + 7, 0.0,
                            gui::color::rgba( {std::min( 1., (double)m_ref_phen( i, j ) ), 0., 0.,0.75} ) ) );
        }
    std::size_t i = m_pos_nest.first;
    std::size_t j = m_pos_nest.second;
    m_p_phen.emplace_back(gui::point( h * j + 1, h * i + 1, 0.0,gui::color::rgba( {0., 1., 0.} ) ),
                gui::point( h * j + 7, h * i + 1, 0.0, gui::color::rgba( {0., 1., 0.,0.75} ) ),
                gui::point( h * j + 7, h * i + 7, 0.0, gui::color::rgba( {0., 1., 0.,0.75} ) ),
                gui::point( h * j + 1, h * i + 7, 0.0, gui::color::rgba( {0., 1., 0.,0.75} ) ) );
    gui::draw( m_p_phen );
    // Affichage des fourmis dans cadran en haut à gauche :
    std::vector< gui::quad > m_p_ants;
    m_p_ants.reserve( m_ref_ants.size( ) );
    for ( auto& ant : m_ref_ants ) {
        const position_t& pos_ant = ant.get_position( );
        m_p_ants.emplace_back( gui::point(h * pos_ant.second + 2, h * pos_ant.first + 2, 0.,
                               gui::color::rgba( {0, 255, 255, 128} ) ),
        gui::point(h * pos_ant.second + 2, h * pos_ant.first + 6, 0.,
                               gui::color::rgba( {0, 255, 255, 128} ) ),
        gui::point(h * pos_ant.second + 6, h * pos_ant.first + 6, 0.,
                               gui::color::rgba( {0, 255, 255, 128} ) ),
        gui::point(h * pos_ant.second + 6, h * pos_ant.first + 2, 0.,
                               gui::color::rgba( {0, 255, 255, 128} ) ));
    }
    gui::draw( m_p_ants );
    // Affichage de la courbe d'enfouragement :
    m_curve.push_back( compteur );
    if ( m_curve.size( ) > 1 ) {
        std::vector< gui::segment > m_p_curve;
        int                         sz_win = m_ref_win.size( ).first;
        int                         ydec   = m_ref_win.size( ).second - 1;
        m_p_curve.reserve( sz_win );
        double h_max_val = 256. / std::max( double( m_curve.back( ) ), 1. );
        double step      = sz_win / (double)( m_curve.size( ) );
        if ( step > 1. ) {
            for ( std::size_t i = 0; i < m_curve.size( ) - 1; i++ ) {
                m_p_curve.emplace_back(
                    gui::point( i * step, ydec - m_curve[i] * h_max_val, 0., gui::color::rgba( {255, 255, 127} ) ),
                    gui::point( ( i + 1 ) * step, ydec - m_curve[i + 1] * h_max_val, 0.,
                                gui::color::rgba( {255, 255, 127} ) ) );
            }
        } else {
            double inv_step = 1. / ( step + 0.001 );
            for ( int i = 0; i < sz_win; ++i ) {
                int ind  = int( i * inv_step );
                int ind2 = int( ( i + 1 ) * inv_step );
                m_p_curve.emplace_back(
                    gui::point( i, ydec - m_curve[ind] * h_max_val, 0., gui::color::rgba( {255, 255, 127} ) ),
                    gui::point( i + 1, ydec - m_curve[ind2] * h_max_val, 0., gui::color::rgba( {255, 255, 127} ) ) );
            }
        }
        gui::draw( m_p_curve );
    }
}
#ifndef _LABYRINTHE_HPP_
# define _LABYRINTHE_HPP_
# include <vector>
# include <utility>
# include "basic_types.hpp"

class labyrinthe
{
public:
    enum direction {
        NORTH=1, EAST=2, SOUTH=4, WEST=8
    };
    labyrinthe( const dimension_t& dims );

    const dimension_t& dimensions() const { return m_dims; }

    unsigned char operator () ( const position_t& ind ) const 
    {
        return m_laby_data[comp_flat_index(ind)];
    }
private:
    std::size_t comp_flat_index(const position_t& ind) const {
        return ind.first*m_dims.second + ind.second;
    }
    dimension_t m_dims;
    std::vector<unsigned char> m_laby_data;
};

#endif
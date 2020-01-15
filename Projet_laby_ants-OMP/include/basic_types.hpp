#ifndef _BASIC_TYPES_HPP_
#define _BASIC_TYPES_HPP_
# include <utility>

using position_t=std::pair<std::size_t,std::size_t>;
inline bool operator == ( const position_t& pos1, const position_t& pos2 )
{
    return (pos1.first == pos2.first ) and (pos1.second == pos2.second);
}

inline bool operator != ( const position_t& pos1, const position_t& pos2 )
{
    return (pos1.first != pos2.first ) or (pos1.second != pos2.second);
}
using dimension_t=std::pair<std::size_t,std::size_t>;

const double h_scal = 8.;

#endif
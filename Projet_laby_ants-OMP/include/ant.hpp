// ant.hpp
#ifndef _ANT_HPP_
# define _ANT_HPP_
# include <utility>
# include "pheronome.hpp"
# include "labyrinthe.hpp"
# include "basic_types.hpp"

class ant
{
public:
    /**
     * Une fourmi peut être dans deux états possibles : chargée ( elle porte de la nourriture ) ou non chargée
     */
    enum state { unloaded = 0, loaded = 1 };
    ant(const position_t& pos, std::size_t m_life) : m_state(unloaded), m_path_to_nest(), max_life(m_life), cur_life(m_life)
    { m_path_to_nest.reserve(5000); m_path_to_nest.push_back(pos); } 
    ant(const ant& a) = default;
    ant( ant&& a ) = default;
    ~ant() = default;

    void set_loaded() { m_state = loaded; }
    void unset_loaded() { m_state = unloaded; }

    bool is_loaded() const { return m_state == loaded; }
    const position_t& get_position() const { return m_path_to_nest.back(); }
    position_t return_to_nest() { position_t p = m_path_to_nest.back(); m_path_to_nest.pop_back(); return p; }
    static void set_exploration_coef(double eps) { m_eps = eps; }

    void advance( pheronome& phen, const labyrinthe& land,
                  const position_t& pos_food, const position_t& pos_nest, std::size_t& cpteur_food );

private:
    static double m_eps; // Coefficient d'exploration commun à toutes les fourmis.
    state m_state;
    std::vector<position_t> m_path_to_nest;
    std::size_t max_life, cur_life;
};

#endif
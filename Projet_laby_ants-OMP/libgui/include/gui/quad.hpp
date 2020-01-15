/**
 * @brief Draw a triangle on the screen
 */
#ifndef _GUI_QUAD_HPP_
#define _GUI_QUAD_HPP_

# include <GL/gl.h>
# include "gui/point.hpp"

namespace gui
{
    /**
     * @brief      Description of a line segment
     */
    class quad
    {
    public:
        quad( const point& pt1, const point& pt2, const point& pt3, const point& pt4 ) : m_pt1(pt1), m_pt2(pt2), m_pt3(pt3), m_pt4(pt4) {}

        GLenum gl_kind() { return GL_QUADS; }
        void emit() const { m_pt1.emit(); m_pt2.emit(); m_pt3.emit(); m_pt4.emit(); }
    private:
        point m_pt1, m_pt2, m_pt3, m_pt4;
    };
}


#endif
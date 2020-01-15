/**
 * @brief Draw a triangle on the screen
 */
#ifndef _GUI_TRIANGLE_HPP_
#define _GUI_TRIANGLE_HPP_

# include <GL/gl.h>
# include "gui/point.hpp"

namespace gui
{
    /**
     * @brief      Description of a line segment
     */
    class triangle
    {
    public:
        triangle( const point& pt1, const point& pt2, const point& pt3 ) : m_pt1(pt1), m_pt2(pt2), m_pt3(pt3) {}

        GLenum gl_kind() { return GL_TRIANGLES; }
        void emit() const { m_pt1.emit(); m_pt2.emit(); m_pt3.emit(); }
    private:
        point m_pt1, m_pt2, m_pt3;
    };
}


#endif
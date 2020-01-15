/**
 * @brief Point to draw on the screen
 */
#ifndef _GUI_POINT_HPP_
# define _GUI_POINT_HPP_
# include <GL/gl.h>
# include "gui/colors.hpp"

namespace gui
{
	/**
	 * @brief      Description of a point
	 */
	class point
	{
	public:
		point( double x, double y, double z = 0., color::rgba col={1.,1.,1.,1.} ) : m_px(x), m_py(y), m_pz(z), m_color(col) {}
        point( const point& pt ) = default;

        point& operator = ( const point& pt ) = default;		
		void set_color( const color::rgba& col ) { m_color = col; }
		void set_color( const color::hsva& col ) { m_color = color::rgba(col); }
		GLenum gl_kind() { return GL_POINTS; }
		void emit() const { glColor4f(float(m_color.r),float(m_color.g),float(m_color.b), float(m_color.a)); glVertex3f(float(m_px),float(m_py),float(m_pz)); }
	private:
		double m_px, m_py, m_pz;
		color::rgba m_color;
	};
}
#endif
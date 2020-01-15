/**
 * @brief Draw a segment on the screen
 */
#ifndef _GUI_SEGMENT_HPP_
#define _GUI_SEGMENT_HPP_

# include <GL/gl.h>
# include "gui/point.hpp"

namespace gui
{
	/**
	 * @brief      Description of a line segment
	 */
	class segment
	{
	public:
		segment( const point& pt_beg, const point& pt_end ) : m_pt_beg(pt_beg), m_pt_end(pt_end) {}

		GLenum gl_kind() { return GL_LINES; }
		void emit() const { m_pt_beg.emit(); m_pt_end.emit(); }
	private:
		point m_pt_beg, m_pt_end;
	};
}


#endif
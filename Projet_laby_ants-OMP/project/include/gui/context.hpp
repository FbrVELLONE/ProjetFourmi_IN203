/**
 * @brief Gui context and factory
 */
#ifndef _GUI_CONTEXT_HPP_
#define _GUI_CONTEXT_HPP_
# include <memory>
# include "gui/window.hpp"

namespace gui
{
	class context
	{
	public:
		context( int& nargs, char* argv[] );
		/**
		 * @brief      Create a new window ( a.k.a screen ) to display with OpenGL acceleration
		 *
		 * This version create a screen with the best resolution found from the hardware screen resolution.
		 *
		 * @return     The new screen
		 */
		window& new_window();
		/**
		 * @brief      Return a new window to display with openGL acceleration
		 *
		 * @param[in]  width           The width for the screen
		 * @param[in]  height          The height for the screen
		 * @param[in]  is_full_screen  Indicates if full screen
		 *
		 * @return     The new screen
		 */
		window& new_window( size_t width, size_t height, bool is_full_screen = false );
		/**
		 * @brief      Return a new window to display with specified title
		 *
		 *  This window is completly defined by the user
		 *
		 * @param[in]  title           The title
		 * @param[in]  width           The width
		 * @param[in]  height          The height
		 * @param[in]  is_full_screen  Indicates if full screen
		 *
		 * @return     A reference on the new window
		 */
		window& new_window( const std::string& title, size_t width, size_t height, bool is_full_screen = false );
	private:
		struct implementation;
		std::shared_ptr<implementation> m_shared_implementation;
	};
}

#endif
#ifndef _GUI_WINDOW_HPP_
#define _GUI_WINDOW_HPP_
# include <string>
# ifdef __APPLE__
#   include <GLUT/glut.h>
# else
#   include <GL/glut.h>
# endif

namespace gui
{
    /**
     * @brief      Graphic window on screen
     * 
     * Create a window to display on a screen. Size, title and fullscreen are parametables.
     */
    class window
    {
    public:
        /**
         * @brief      Create a default window
         * 
         * The default window : title : "Default window", size : (800,800), isFullScreen = false
         */        
        window();
        /**
         * @brief      Create a window with a specified  title
         * 
         * This window : size : (800,800), isFullScreen = false
         *
         * @param[in]  title  The title of the window
         */
        window(const std::string& title);
        /**
         * @brief      Create a window with a specified title
         *
         * @param[in]  title         The title of the window
         * @param[in]  size_x        The size in x of the window
         * @param[in]  size_y        The size in y of the window
         * @param[in]  isFullScreen  Indicates if full screen
         */
        window(const std::string& title, int size_x, int size_y, bool isFullScreen);
        window(const window& w) = delete;
        window(window&& w) = default;
        ~window();

        /**
         * @brief      Return the size in pixels in each direction of the window
         *
         * @return     A pair of int : first : number of pixels in x direction, second : number of pixels in y direction
         */
        std::pair<int,int> size() const { return {m_size_x,m_size_y}; }

        /**
         * @brief      Clear the window
         */
        void clear() { glClear ( GL_COLOR_BUFFER_BIT ); }
        /**
         * @brief      Put the ready graphics on the window.
         */
        void blit() { glutSwapBuffers(); }// Force le swapping pour librairie opengl

    private:
        int m_size_x, m_size_y;
        bool m_is_full_screen;
        int m_num_window;
    };
    /**
     * @brief      Draw some shapes on opengl display
     *
     * @param[in]  pts        The shapes to draw
     *
     * @tparam     Container  containing the shapes ( same kind of shapes )
     */
    template<typename Container> void draw( const Container& pts )
    {
        auto v = (*std::begin(pts));
        glBegin(v.gl_kind());
        for ( auto it = std::begin(pts); it != std::end(pts); ++it ) (*it).emit();
        glEnd();
     }
}

#endif
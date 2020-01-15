 # include "gui/window.hpp"

namespace gui
{
    window::window()
    {
        glutInitWindowSize(800, 800);
        glutCreateWindow("Default window");
        m_size_x = glutGet(GLUT_WINDOW_WIDTH);
        m_size_y = glutGet(GLUT_WINDOW_HEIGHT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0f, m_size_x, m_size_y, 0.0f, 0.0f, 1.0f);
    }
    // ---------------------------------------------------------------------    
    window::window(const std::string& title)
    {
        glutCreateWindow(title.c_str());
        m_size_x = glutGet(GLUT_WINDOW_WIDTH);
        m_size_y = glutGet(GLUT_WINDOW_HEIGHT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0f, m_size_x, m_size_y, 0.0f, 0.0f, 1.0f);
    }
    // ---------------------------------------------------------------------    
    window::window(const std::string& title, int size_x, int size_y, bool isFullScreen) :
        m_size_x(size_x), m_size_y(size_y)
    {
        glutInitWindowSize(size_x, size_y);
        glutCreateWindow(title.c_str());
        if (isFullScreen)
            glutFullScreen();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0f, size_x, size_y, 0.0f, 0.0f, 1.0f);
    }
    // ---------------------------------------------------------------------    
    window::~window() { glutDestroyWindow(m_num_window); }
}
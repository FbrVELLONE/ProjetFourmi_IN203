#include "gui/context.hpp"
#include <vector>
#include <sstream>
#include <stdexcept>

namespace gui {
struct context::implementation {
	using list_of_windows = std::vector<window>;
    implementation( int& nargs, char* vargs[] );
    ~implementation();
    static std::shared_ptr<context::implementation> m_unique_instance;
    static bool is_initialized;
    window& add_window();
    window& add_window( size_t width, size_t height, bool is_full_window );
    window& add_window( const std::string& title, size_t width, size_t height, bool is_full_window );
    list_of_windows m_lst_windows;
};
// ---------------------------------------------------------------------------------
std::shared_ptr<context::implementation> context::implementation::m_unique_instance{};
// ---------------------------------------------------------------------------------
bool context::implementation::is_initialized = false;
// ---------------------------------------------------------------------------------
context::implementation::implementation(int& nargs, char* vargs[]) : m_lst_windows() {
    if (is_initialized == false) {
        glutInit(&nargs, vargs);
        glutInitDisplayMode(GLUT_DOUBLE|GLUT_STENCIL);
        is_initialized = true;
    }
}
// ---------------------------------------------------------------------------------
context::implementation::~implementation()
{}
// ---------------------------------------------------------------------------------
window&
context::implementation::add_window()
{
	m_lst_windows.emplace_back();
	return m_lst_windows.back();
}
// ---------------------------------------------------------------------------------
window&
context::implementation::add_window( size_t width, size_t height, bool is_full_window )
{
	m_lst_windows.emplace_back("Display window", width, height, is_full_window);
	return m_lst_windows.back();
}
// ---------------------------------------------------------------------------------
window&
context::implementation::add_window( const std::string& title, size_t width, size_t height, bool is_full_window )
{
    m_lst_windows.emplace_back(title.c_str(), width, height, is_full_window);
    return m_lst_windows.back();
}
// ---------------------------------------------------------------------------------
context::context(int& nargs, char* vargs[]) : m_shared_implementation(std::make_shared<typename context::implementation>(nargs,vargs)) 
{}
// ---------------------------------------------------------------------------------
window& context::new_window()
{
	return m_shared_implementation->add_window();
}
// ---------------------------------------------------------------------------------
window& context::new_window(size_t width, size_t height, bool is_full_window)
{
	return m_shared_implementation->add_window( width, height, is_full_window);
}
// ---------------------------------------------------------------------------------
window& context::new_window(const std::string& title, size_t width, size_t height, bool is_full_window)
{
    return m_shared_implementation->add_window( title.c_str(), width, height, is_full_window);
}
// ---------------------------------------------------------------------------------
}
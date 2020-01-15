#include "gui/event_manager.hpp"
#include <iostream>

namespace {
    std::function<void(void)> *pt_display = nullptr, *pt_idle = nullptr;
    void display() {
        if (pt_display != nullptr) {
            (*pt_display)();
        }
    }
    void idle() {
        if (pt_idle != nullptr) {
            (*pt_idle)();
        }
    }
    gui::event_manager* pt_manager = nullptr;
    void key_handler( unsigned char key, int x, int y)
    {
        if (pt_manager != nullptr ) {
            pt_manager->key_event(int(key));
        }
    }
}

namespace gui {
        event_manager::event_manager() : on_display_trigger{[] (){} }, on_idle_trigger{[](){} }, key_trigger{}, 
         mouse_button_trigger{}, mouse_motion_trigger{[] (int,int) {} }
                                         
        {
            glutDisplayFunc(display);
            glutIdleFunc(idle);
            glutKeyboardFunc(key_handler);
            pt_manager = this;
            pt_display = &on_display_trigger;
            pt_idle    = &on_idle_trigger;
            // Regarde pour rajouter les événements souris
        }
        //
        void event_manager::loop() {
            glutMainLoop();
        }
}
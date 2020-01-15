/**
 * @brief Event manager : to attach processus 
 */
#ifndef _GUI_EVENT_MANAGER_HPP_
# define _GUI_EVENT_MANAGER_HPP_
# include <functional>
# include <list>
# include <utility>
# include <map>
# ifdef __APPLE__
#   include <GLUT/glut.h>
# else
#   include <GL/glut.h>
# endif

namespace gui
{
	/**
	 * @brief      Manager for events 
	 */
	class event_manager
	{
	public:
		event_manager();
		/**
		 * @brief      Défini une fonction à déclencher lors d'un événement pour une touche particulière du clavier
		 *
		 * @param[in]  key_code  Le code de la touche clavier ( en  ASCII )
		 * @param[in]  func      La fonction à déclencher
		 *
		 * @tparam     F         Fonction prenant un paramètre indiquant si la touche est pressée ou relachée
		 */
        template<typename F>
        void on_key_event( int key_code, F func ) {
            key_trigger[key_code] = std::function<void(int)>{func};
        }
        /**
         * @brief      Trigger the event relative of the key code
         *
         * @param[in]  key_code  The key code
         */
        void key_event(int key_code) const {
            auto it = key_trigger.find(key_code);
            if (it != key_trigger.end())
                (*it).second(key_code);
        }
        /**
         * @brief      Défini une fonction associée à un événement sur un bouton de la souris
         *
         * @param[in]  num_but  le numéro du bouton
         * @param[in]  func     la fonction appelée lors de l'événement. Cette fonction prend trois
         *                      paramètres entiers : 1. Si le bouton a été enfoncé ( 1 ) ou relaché ( 0 )
         *                                           2. et 3. : Les coordonnées de la souris sur l'écran
         *
         * @tparam     F        Une fonction prenant trois entiers en paramètre et ne renvoyant rien.
         */
        template<typename F>
        void on_mouse_button_event( int num_but, F func ) {
            mouse_button_trigger[num_but] = std::function<void(int,int,int)>{func};
        }
        /**
         * @brief      Trigger the mouse button event relative to the button number
         *
         * @param[in]  num_but  The number but
         * @param[in]  x        Abcisse position of the mouse
         * @param[in]  y        Ordinate position of the mouse
         */
        void mouse_button_event(int num_but, int x, int y) const
        {
            auto it = mouse_button_trigger.find(num_but);
            if (it != mouse_button_trigger.end()) {
                (*it).second(num_but, x, y);
            }
        }
        /**
         * @brief      Défini une fonction associée au mouvement de la souris sur l'écran
         *
         * @param[in]  func  La fonction à déclencher lorsqu'on bouge la souris
         *
         * @tparam     F     Une fonction prenant deux paramètres entier donnant la position relative de la souris
         *                   par rapport à son ancienne position.
         */
        template<typename F>
        void on_mouse_motion( F func ) {
            mouse_motion_trigger = std::function<void(int,int)>{func};
        }
        /**
         * @brief      Trigger the event relative of the displacement of the mouse
         *
         * @param[in]  x     Delta move of the mouse along the abcisse
         * @param[in]  y     Delta move of the mouse along the ordinate
         */
        void mouse_motion_event(int x, int y) {
            mouse_motion_trigger(x,y);
        }
        /**
         * @brief Défini une fonction d'affichage lors d'un rafraichissement de la fenêtre
         * 
         * @param[in] func La fonction d'affichage ( ne prend rien en paramêtre, et ne retourne rien... )
         */
        template<typename F>
        void on_display( F func ) {
            on_display_trigger = std::function<void(void)>{func};
        }
        /**
         * @brief Défini une fonction d'exécution quand le programme ne fait rien d'autre ( permet les animations )
         * 
         * @param[in] func La fonction à exécuter lorsqu'on ne fait rien d'autre ( ne prend rien en paramêtre, et ne retourne rien... )
         */
        template<typename F>
        void on_idle( F func ) {
            on_idle_trigger = std::function<void(void)>{func};
        }

        /**
         * @brief      Do an infinite loop for interface
         */
        void loop();
	private:
        std::function<void(void)> on_display_trigger, on_idle_trigger;
        std::map<unsigned char,std::function<void(int)>> key_trigger;
        std::map<int,std::function<void(int,int,int)>> mouse_button_trigger;
        std::function<void(int,int)> mouse_motion_trigger;
	};
}

#endif
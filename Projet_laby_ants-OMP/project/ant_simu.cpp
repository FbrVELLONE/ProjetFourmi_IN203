#include <vector>
#include <iostream>
#include <random>
#include "labyrinthe.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
# include "gui/context.hpp"
# include "gui/colors.hpp"
# include "gui/point.hpp"
# include "gui/segment.hpp"
# include "gui/triangle.hpp"
# include "gui/quad.hpp"
# include "gui/event_manager.hpp"
# include "display.hpp"
#include <chrono>
#include <mpi.h>
#include <fstream>
#include <cstdlib> 

void advance_time( const labyrinthe& land, pheronome& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur )
{
        //Horloge adv
    std::chrono::time_point<std::chrono::system_clock> startAdv, endAdv;
    startAdv = std::chrono::system_clock::now();

    for ( size_t i = 0; i < ants.size(); ++i )
            ants[i].advance(phen, land, pos_food, pos_nest, cpteur);

        endAdv = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = endAdv - startAdv;
        std::cout << "Advance: " << duration.count() <<  std::endl;

    phen.do_evaporation();
    
    phen.update();
}

int main(int nargs, char* argv[])
{
    
    const dimension_t dims{32, 64};// Dimension du labyrinthe
    const std::size_t life = int(dims.first*dims.second);
    const int nb_ants = 2*dims.first*dims.second; // Nombre de fourmis
    const double eps = 0.75;  // Coefficient d'exploration
    const double alpha=0.97; // Coefficient de chaos
    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta=0.999; // Coefficient d'évaporation
    labyrinthe laby(dims);
    // Location du nid
    position_t pos_nest{dims.first/2,dims.second/2};
    // Location de la nourriture
    position_t pos_food{dims.first-1,dims.second-1};
                          
        //Timers
    std::chrono::time_point<std::chrono::system_clock> victStart, victEnd;
    victStart = std::chrono::system_clock::now();

    int rank, state;
    // Initialisation de MPI
    MPI_Init (&nargs , &argv);

    // Lit le nombre de tâches
    MPI_Comm_size (MPI_COMM_WORLD , &state);

    // Lit mon rang
    MPI_Comm_rank (MPI_COMM_WORLD , &rank);

    const int buffer_size = 1 + 2*nb_ants + laby.dimensions().first*laby.dimensions().second;

    if (rank == 0){ //thread for only display
        std::vector<ant> ants;
        ants.reserve(nb_ants);
        size_t food_quantity = 0;
        size_t ants_start = 0;
        size_t pher_start = nb_ants*2 + 1;
            // On crée toutes les fourmis dans la fourmilière.
        pheronome phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);

            //Recv les buffers
        std::vector<double> bufferRec(buffer_size);

        MPI_Request request;
        MPI_Status status;
        MPI_Recv(bufferRec.data(), bufferRec.size(), MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, &status);

        food_quantity = bufferRec.back();
        for (size_t i = ants_start; i < pher_start; i+=2){
            ants.emplace_back(position_t(bufferRec[i], bufferRec[i+1]), life);
        }
        phen.copy(std::vector<double>(bufferRec.begin() + pher_start, bufferRec.end()-1));

                //Partie responsable pour l'affichage
        gui::context graphic_context(nargs, argv);
        gui::window& win =  graphic_context.new_window(h_scal*laby.dimensions().second,h_scal*laby.dimensions().first+266);

        display_t displayer( laby, phen, pos_nest, pos_food, ants, win );

        gui::event_manager manager;
        //manager.on_key_event(int('q'), [] (int code) { exit(0); });
        manager.on_display([&] { displayer.display(food_quantity); win.blit(); });
        manager.on_idle([&] () { 
            
            std::chrono::time_point<std::chrono::system_clock> start, end;
            //Start le counter pour le display
            start = std::chrono::system_clock::now();
            
            displayer.display(food_quantity); 

            //Duration
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Display: " << duration.count() << ", thread:" << rank << std::endl;

            if (food_quantity >= 10000){
                victEnd = std::chrono::system_clock::now();
                std::chrono::duration<double> duration = victEnd - victStart;
                std::ofstream outfile ("saida.txt");

                outfile << "Victoire: " << duration.count() << ", thread:" << rank << std::endl;

                outfile.close();
            }

            win.blit(); 

            MPI_Recv(bufferRec.data(), bufferRec.size(), MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, &status);
            food_quantity = bufferRec.back();
            for(size_t i = ants_start, j = 0; i < pher_start; i += 2, ++j){
                ants[j].set_position(position_t(bufferRec[i], bufferRec[i+1]));
            }
            phen.copy(std::vector<double>(bufferRec.begin() + pher_start, bufferRec.end()-1));
        });
        manager.loop();
    }else if (rank == 1){//thread "calcs only"
                    //Initialization des calculs
        // Définition du coefficient d'exploration de toutes les fourmis.
        ant::set_exploration_coef(eps);
        size_t food_quantity = 0;
        // On va créer toutes les fourmis dans le nid :
        std::vector<ant> ants;
        ants.reserve(nb_ants);
        for ( size_t i = 0; i < nb_ants; ++i )
            ants.emplace_back(pos_nest, life);
        // On crée toutes les fourmis dans la fourmilière.
        pheronome phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);

        while (1){
            std::vector<double> bufferSend;

            for (size_t i = 0; i<nb_ants; i++){
                bufferSend.emplace_back((double)ants[i].get_position().first);
                bufferSend.emplace_back((double)ants[i].get_position().second);
            }

            for (size_t i = 0; i < laby.dimensions().first; i++){
                for (size_t j = 0; j < laby.dimensions().second; j++){
                    bufferSend.emplace_back((double)phen(i, j));
                }
            }

            bufferSend.emplace_back((double)food_quantity);

            MPI_Request request;
            MPI_Status status;
            MPI_Isend(bufferSend.data(), bufferSend.size(), MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, &request);
            advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);
            MPI_Wait(&request, &status);
        }
    }

    MPI_Finalize();
    return MPI_SUCCESS;
}
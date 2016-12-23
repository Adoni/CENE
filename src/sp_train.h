//
// Created by Adoni1203 on 16/7/22.
//

#ifndef DLNE_SP_TRAIN_H
#define DLNE_SP_TRAIN_H

#include "network_data.h"
#include <chrono>

using namespace std;
using namespace dynet;
namespace sp_train {
    void RunSingleProcess(DLNEModel *learner, Trainer *trainer, GraphData &graph_data) {
        std::cout << "==================" << std::endl << "START TRAINING" << std::endl << "==================" <<
                  std::endl;
        std::cout << "Single trainer: " << trainer->model->lookup_parameters_list().size() << std::endl;
        int MAX_ITERATION = 100;
        const unsigned report_every_i = graph_data.vv_edgelist.size();
        const unsigned update_epoch_every_i = 10000;
        const int save_every_i = graph_data.vv_edgelist.size() * 10;
        unsigned epoch = 0;
        int total_iteration = 0;
        auto total_start_time = std::chrono::high_resolution_clock::now();

        double loss = 0;
        unsigned pair = 0;

        while (1) {
            for (unsigned i = 0; i < graph_data.vv_edgelist.size(); ++i) {
                Edge edge = graph_data.vv_edgelist[i];
                loss += learner->TrainVVEdge(edge, graph_data);
                trainer->update();

                total_iteration += 1;
                pair += 1;
                if (total_iteration % update_epoch_every_i == 0) {
                    //                trainer->update_epoch();
                    trainer->eta = trainer->eta0 *
                                   (1 - total_iteration / (float) (MAX_ITERATION * graph_data.vv_edgelist.size() + 1));
                    if (trainer->eta < trainer->eta0 * 0.0001) {
                        trainer->eta = trainer->eta0 * 0.0001;
                    }
                }
                if (total_iteration % report_every_i == 0) {
                    auto now_time = std::chrono::high_resolution_clock::now();
                    trainer->status();
                    cerr << " E = " << loss / pair << " ppl=" << exp(loss / pair) << endl;
                    cerr << "Totally using " <<
                         std::chrono::duration<double, std::milli>(now_time - total_start_time).count() / 60000
                         << " min" <<
                         endl;
                    loss = 0;
                    pair = 0;
                }
            }
            epoch++;
            if (epoch > MAX_ITERATION) {
                break;
            }
        }
    }
}
#endif //DLNE_SP_TRAIN_H

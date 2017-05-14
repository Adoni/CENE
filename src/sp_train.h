//
// Created by Adoni1203 on 16/7/22.
//

#ifndef DLNE_SP_TRAIN_H
#define DLNE_SP_TRAIN_H

#include "network_data.h"
#include "network_embedding.h"
#include <chrono>

using namespace std;
using namespace dynet;
namespace sp_train {
    void RunSingleProcess(DLNEModel *learner, Trainer *params_trainer,
                          NetworkData &network_data, unsigned num_iterations,
                          unsigned save_every_i,
                          unsigned report_every_i, unsigned batch_size, unsigned update_epoch_every_i) {
        std::cout << "==================" << std::endl << "START TRAINING" << std::endl << "==================" <<
                  std::endl;
        std::cout << "Iterations: " << batch_size << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Save every " << save_every_i << "iterations" << std::endl;
        report_every_i = report_every_i / batch_size;
        std::cout << "Report every " << report_every_i << "batches" << std::endl;

        std::vector<unsigned> train_indices(network_data.edge_list.size());
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::shuffle(train_indices.begin(), train_indices.end(), (*dynet::rndeng));
        std::vector<unsigned>::iterator begin = train_indices.begin();

        for (unsigned iter = 0; iter < num_iterations; ++iter) {
            unsigned batch_count = 0;
            float loss = 0.0;
            while (begin != train_indices.end()) {
                std::vector<unsigned>::iterator end = begin + batch_size;
                if (end > train_indices.end()) {
                    end = train_indices.end();
                }
                for (auto instance = begin; instance < end; instance++) {
                    learner->Train(network_data.edge_list[*instance], network_data);
                    params_trainer->update();
                }
                batch_count++;
                if (batch_count % report_every_i == 0) {
                    std::cout << "Eta = " << params_trainer->eta << "\tloss = " << loss << std::endl;
                    loss = 0.0;
                }
                if (batch_count % update_epoch_every_i == 0) {
                    params_trainer->update_epoch();
                }
                begin = end;
            }
            std::shuffle(train_indices.begin(), train_indices.end(), (*dynet::rndeng));
            begin = train_indices.begin();

            if (iter % save_every_i == 0) {
                std::ostringstream ss;
                ss << learner->get_learner_name() << "_embedding_pid" << getpid() << "_alpha_";
                for (auto alpha:learner->alpha) {
                    ss << std::setprecision(2) << alpha << "_";
                }
                ss << unsigned(iter / save_every_i) << ".data";
                learner->SaveEmbedding(ss.str(), "relation.data", network_data);
            }

        }
    }
}
#endif //DLNE_SP_TRAIN_H

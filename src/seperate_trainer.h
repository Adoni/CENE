//
// Created by Adoni1203 on 16/7/25.
//

#ifndef DLNE_SEPERATE_TRAINER_H
#define DLNE_SEPERATE_TRAINER_H

#include "cnn/training.h"
#include <vector>
#include "cnn/model.h"
#include "cnn/shadow-params.h"

using namespace cnn;
struct SeperateSimpleSGDTrainer : public Trainer {
    explicit SeperateSimpleSGDTrainer(Model* m, real lam = 1e-6, real e0 = 0.1) : Trainer(m, lam, e0) {}
    void update(real scale) override;
    void update(const std::vector<LookupParameters*> &lookup_params, const std::vector<Parameters*> &params, real scale = 1);
    void update_params(real scale = 1.0);
    void update_lookup_params(real scale = 1.0);
    void update_params(const std::vector<Parameters*> &params, real scale = 1);
    void update_lookup_params(const std::vector<LookupParameters*> &lookup_params, real scale = 1);
};

#endif //DLNE_SEPERATE_TRAINER_H

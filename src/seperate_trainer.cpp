//
// Created by Adoni1203 on 16/7/25.
//

#include "seperate_trainer.h"

void SeperateSimpleSGDTrainer::update(real scale) {
    update(model->lookup_parameters_list(), model->parameters_list(), scale);
}

void SeperateSimpleSGDTrainer::update(const std::vector<LookupParameters*> &lookup_params, const std::vector<Parameters*> &params, real scale) {
    const float gscale = clip_gradients();
    for (auto p : params) {
#if HAVE_CUDA
        gpu::sgd_update(p->values.d.size(), p->g.v, p->values.v, eta * scale * gscale, lambda);
#else
        auto reg = (*p->values) * lambda;
        *p->values -= ((eta * scale * gscale) * *p->g + reg);
#endif
        p->clear();
    }
    for (auto p : lookup_params) {
        for (auto i : p->non_zero_grads) {
#if HAVE_CUDA
            gpu::sgd_update(p->values[i].d.size(), p->grads[i].v, p->values[i].v, eta * scale * gscale, lambda);
#else
            auto reg = (*p->values[i]) * lambda;
            *p->values[i] -= (*p->grads[i] * (eta * scale * gscale) + reg);
#endif
        }
        p->clear();
    }
    ++updates;
}


void SeperateSimpleSGDTrainer::update_params(real scale) {
    update_params(model->parameters_list(), scale);
}

void SeperateSimpleSGDTrainer::update_params(const std::vector<Parameters*> &params, real scale) {
    const float gscale = clip_gradients();
    for (auto p : params) {
#if HAVE_CUDA
        gpu::sgd_update(p->values.d.size(), p->g.v, p->values.v, eta * scale * gscale, lambda);
#else
        auto reg = (*p->values) * lambda;
        *p->values -= ((eta * scale * gscale) * *p->g + reg);
#endif
        p->clear();
    }
    ++updates;
}

void SeperateSimpleSGDTrainer::update_lookup_params(real scale) {
    update_lookup_params(model->lookup_parameters_list(), scale);
}


void SeperateSimpleSGDTrainer::update_lookup_params(const std::vector<LookupParameters *> &lookup_params, real scale){
    const float gscale = clip_gradients();
    for (auto p : lookup_params) {
        for (auto i : p->non_zero_grads) {
#if HAVE_CUDA
            gpu::sgd_update(p->values[i].d.size(), p->grads[i].v, p->values[i].v, eta * scale * gscale, lambda);
#else
            auto reg = (*p->values[i]) * lambda;
            *p->values[i] -= (*p->grads[i] * (eta * scale * gscale) + reg);
#endif
        }
        p->clear();
    }
    ++updates;
}
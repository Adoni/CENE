//
// Created by Adoni1203 on 2017/2/5.
//


#include "L2SGD.h"
#include <boost/serialization/vector.hpp>

#include "dynet/io-macros.h"

// Macros for defining parameter update functions
#ifdef __CUDACC__
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real scale, real gscale, const std::vector<Tensor*> & values);
#elif defined(HAVE_CUDA)
// This is correct, but dying when models are read and written.
// if(values[0]->device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)values[0]->device,scale,gscale,values); }
// else if(values[0]->device->type == DeviceType::GPU) { update_rule_dev(*(Device_GPU*)values[0]->device,scale,gscale,values); }
// else { abort(); }
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  extern template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real scale, real gscale, const std::vector<Tensor*> & values) { \
    if(default_device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)default_device,scale,gscale,values); } \
    else if(default_device->type == DeviceType::GPU) { update_rule_dev(*(Device_GPU*)default_device,scale,gscale,values); } \
    else { abort(); } \
  }
#else
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real scale, real gscale, const std::vector<Tensor*> & values) { \
    if(default_device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)default_device,scale,gscale,values); } \
    else { throw std::runtime_error("Bad device in MyTrainer::update_rule"); } \
  }
#endif

namespace dynet {
// Perform update of ts[0]=parameters, ts[1]=gradients
template<class MyDevice>
void L2SimpleSGDTrainer::update_rule_dev(const MyDevice &dev,
                                         real scale,
                                         real gscale,
                                         const std::vector<Tensor *> &ts) {
    ts[0]->tvec().device(*dev.edevice) -= ts[0]->tvec() * l2lambda;
    ts[0]->tvec().device(*dev.edevice) -=
        ts[1]->tvec() * (eta * scale * gscale / model->weight_decay.current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(L2SimpleSGDTrainer)

#ifndef __CUDACC__
void L2SimpleSGDTrainer::update_params(real scale, real gscale, size_t idx) {
    auto &p = model->parameters_list()[idx];
    update_rule(scale, gscale, {&p->values, &p->g});
}
void L2SimpleSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
    auto &p = model->lookup_parameters_list()[idx];
    update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx]});
}
void L2SimpleSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
    auto &p = model->lookup_parameters_list()[idx];
    update_rule(scale, gscale, {&p->all_values, &p->all_grads});
}
#endif

template<class Archive>
void L2SimpleSGDTrainer::serialize(Archive &ar, const unsigned int) {
    ar & boost::serialization::base_object<Trainer>(*this);
}
DYNET_SERIALIZE_IMPL(L2SimpleSGDTrainer)
}
//
// Created by Adoni1203 on 2017/2/5.
//

#ifndef DLNE_L2SGD_H_H
#define DLNE_L2SGD_H_H

#include "dynet/training.h"
#include "dynet/globals.h"
#include <vector>

#include <boost/serialization/export.hpp>

#include "dynet/model.h"
#include "dynet/shadow-params.h"
#include "dynet/io-macros.h"

#define DYNET_TRAINER_DEFINE_DEV_IMPL() \
  void update_params(real scale, real gscale, size_t idx) override; \
  void update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) override; \
  void update_lookup_params(real scale, real gscale, size_t idx) override; \
  template <class MyDevice> \
  void update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  void update_rule(real scale, real gscale, const std::vector<Tensor*> & values) override;


namespace dynet {
struct L2SimpleSGDTrainer : public Trainer {
  /**
   * \brief Constructor
   *
   * \param m Model to be trained
   * \param e0 Initial learning rate
   * \param edecay Learning rate decay parameter.
   */
  explicit L2SimpleSGDTrainer(Model &m, dynet::real e0 = 0.1, dynet::real edecay = 0.0, dynet::real l2lambda = 1e-6)
      : Trainer(m, e0, edecay), l2lambda(l2lambda) {}
 protected:
  DYNET_TRAINER_DEFINE_DEV_IMPL()
 private:
  L2SimpleSGDTrainer() {}
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int);
  dynet::real l2lambda;
};
}
#endif //DLNE_L2SGD_H_H
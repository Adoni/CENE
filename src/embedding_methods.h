//
// Created by Adoni1203 on 16/7/20.
//

#ifndef DLNE_EMBEDDING_METHODS_H
#define DLNE_EMBEDDING_METHODS_H

#include "network_data.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "dynet/gru.h"
#include "dynet/dict.h"
#include "dynet/training.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>


using namespace dynet;

class ContentEmbeddingMethod {
public:
    virtual ~ContentEmbeddingMethod() {}

    virtual Expression get_embedding(const CONTENT_TYPE &content, ComputationGraph &cg) = 0;

    std::string get_method_name() {
        return method_name;
    }

    void initial_look_up_table_from_file(std::string file_name, Dict &d) {
        std::cout << "Initializing lookup table from " << file_name << " ..." << std::endl;
        std::string UNK = "<unk>";
        std::ifstream em_in(file_name);
        assert(em_in);
        unsigned em_count, em_size;
        unsigned unknow_id = d.convert(UNK);
        em_in >> em_count >> em_size;
        assert(em_size == W_EM_DIM);
        std::vector<float> e(em_size);
        std::string w;
        unsigned initialized_word_count = 0;
        for (unsigned i = 0; i < em_count; i++) {
            em_in >> w;
            for (unsigned j = 0; j < em_size; j++) {
                em_in >> e[j];
            }
            unsigned index = d.convert(w);
            if (index == unknow_id) continue;
            initialized_word_count++;
            assert(index < d.size() && index >= 0);
            p.initialize(index, e);
        }
        std::cout << "Initialize " << initialized_word_count << " words" << std::endl;
        std::cout << d.size() - initialized_word_count << " words not initialized" << std::endl;
    }

    void initial_look_up_table(unsigned lookup_table_size) {
        std::vector<float> init(W_EM_DIM);
        std::uniform_real_distribution<> dis(-0.5, 0.5);
        for (unsigned i = 0; i < lookup_table_size; i++) {
            for (unsigned j = 0; j < init.size(); j++) {
                init[j] = (float) dis(*dynet::rndeng) / W_EM_DIM;
            }
            p.initialize(i, init);
        }
    }

    LookupParameter p;
    unsigned W_EM_DIM;
    unsigned C_EM_DIM;
    std::string method_name;
    bool use_const_lookup;
};

class WordAvg_CE : public ContentEmbeddingMethod {
public:
    explicit WordAvg_CE(
            Model &params_model,
            unsigned word_embedding_size,
            unsigned content_embedding_size,

            Dict &d) {
        assert(word_embedding_size == content_embedding_size);
        this->W_EM_DIM = word_embedding_size;
        this->C_EM_DIM = content_embedding_size;
        this->method_name = "WordAvg";
        this->use_const_lookup = use_const_lookup;
        p = params_model.add_lookup_parameters(d.size(), {(unsigned) W_EM_DIM});
        initial_look_up_table(d.size());
    }

    ~WordAvg_CE() {}

    Expression get_embedding(const CONTENT_TYPE &content, ComputationGraph &cg) {
        std::vector<Expression> all_word_embedding;
        for (unsigned i = 0; i < content.size(); i++) {
            std::vector<Expression> sentence_expression;
            for (unsigned j = 0; j < content[i].size(); j++) {
                if (use_const_lookup) {
                    sentence_expression.push_back(const_lookup(cg, p, content[i][j]));
                } else {
                    sentence_expression.push_back(lookup(cg, p, content[i][j]));
                }
            }
            all_word_embedding.push_back(average(sentence_expression));
        }
        return average(all_word_embedding);
    }
};

class GRU_CE : public ContentEmbeddingMethod {
public:
    GRUBuilder builder;

    explicit GRU_CE(
            Model &params_model,
            unsigned word_embedding_size,
            unsigned content_embedding_size,
            Dict &d, std::string language_model_file = "") {
        this->W_EM_DIM = word_embedding_size;
        this->C_EM_DIM = content_embedding_size;
        this->method_name = "GRU";
        this->use_const_lookup = use_const_lookup;

        builder = GRUBuilder(1, W_EM_DIM, C_EM_DIM, params_model);
        if (language_model_file != "") {
            std::cout << "Reading language model parameters from " << language_model_file << "...\n";
            std::ifstream in(language_model_file);
            assert(in);
            boost::archive::binary_iarchive ia(in);
            ia >> params_model;
        }
        p = params_model.add_lookup_parameters(d.size(), {(unsigned)W_EM_DIM});
        initial_look_up_table(d.size());
    }

    ~GRU_CE() {}

    Expression get_embedding(const CONTENT_TYPE &content, ComputationGraph &cg) {
        std::vector<Expression> all_hidden;
        assert(content.size() > 0);
        for (auto s:content) {
            assert(s.size() > 0);
            builder.new_graph(cg);
            builder.start_new_sequence();
            std::vector<Expression> sent;
            for (auto w:s) {
                Expression i_x_t = lookup(cg, p, w);
                sent.push_back(builder.add_input(i_x_t));
            }
            all_hidden.push_back(average(sent));
        }
        return average(all_hidden);
    }
};

class BiGRU_CE : public ContentEmbeddingMethod {
public:
    GRUBuilder builder;

    explicit BiGRU_CE(
            Model &params_model,
            unsigned word_embedding_size,
            unsigned content_embedding_size,

            Dict &d) {
        this->W_EM_DIM = word_embedding_size;
        this->C_EM_DIM = content_embedding_size;
        this->method_name = "BiGRU";
        this->use_const_lookup = use_const_lookup;

        builder = GRUBuilder(1, W_EM_DIM, C_EM_DIM / 2, params_model);
        p = params_model.add_lookup_parameters(d.size(), {W_EM_DIM});
        initial_look_up_table(d.size());
    }

    ~BiGRU_CE() {}

    Expression get_embedding(const CONTENT_TYPE &content, ComputationGraph &cg) {
        std::vector<Expression> all_hidden;
        assert(content.size() > 0);
        for (auto s:content) {
            assert(s.size() > 0);

            builder.new_graph(cg);
            builder.start_new_sequence();
            std::vector<Expression> sent1;

            for (auto w:s) {
                Expression i_x_t = lookup(cg, p, w);
                sent1.push_back(builder.add_input(i_x_t));
            }

            builder.new_graph(cg);
            builder.start_new_sequence();
            std::vector<Expression> sent2;
            for (int i = s.size() - 1; i >= 0; i--) {
                Expression i_x_t = lookup(cg, p, s[i]);
                sent2.push_back(builder.add_input(i_x_t));
            }

            std::vector<Expression> sent;
            for (int i = 0; i < s.size(); i++) {
                sent.push_back(concatenate({sent1[i], sent2[i]}));
            }

            all_hidden.push_back(average(sent));
        }
        return average(all_hidden);
    }
};

class CNN_CE : public ContentEmbeddingMethod {
public:
    explicit CNN_CE(Model &params_model,
                    unsigned word_embedding_size,
                    unsigned content_embedding_size,
                    const std::vector<std::pair<unsigned, unsigned>> &info,
                    Dict &d) :
            zeros(word_embedding_size, 0.), filters_info(info) {
        this->W_EM_DIM = word_embedding_size;
        this->C_EM_DIM = content_embedding_size;
        this->method_name = "CNN";
        p = params_model.add_lookup_parameters(d.size(), {W_EM_DIM});
        initial_look_up_table(d.size());

        unsigned n_filter_types = info.size();
        unsigned combined_dim = 0;
        p_filters.resize(n_filter_types);
        p_biases.resize(n_filter_types);
        for (unsigned i = 0; i < info.size(); ++i) {
            const auto &filter_width = info[i].first;
            const auto &nb_filters = info[i].second;
            p_filters[i].resize(nb_filters);
            p_biases[i].resize(nb_filters);
            for (unsigned j = 0; j < nb_filters; ++j) {
                p_filters[i][j] = params_model.add_parameters({W_EM_DIM, filter_width});
                p_biases[i][j] = params_model.add_parameters({W_EM_DIM});
                combined_dim += W_EM_DIM;
            }
        }
        p_W = params_model.add_parameters({C_EM_DIM, combined_dim});
    }


    dynet::expr::Expression get_embedding(const CONTENT_TYPE &content, ComputationGraph &cg) {
        unsigned len = content.size();
        auto padding = dynet::expr::zeroes(cg, {(unsigned) zeros.size()});
        std::vector<dynet::expr::Expression> s;
        for (auto c:content) {
            for (auto w:c) {
                s.push_back(lookup(cg, p, w));
            }
        }
        assert(s.size() > 0);
        std::vector<dynet::expr::Expression> tmp;
        for (unsigned ii = 0; ii < filters_info.size(); ++ii) {
            const auto &filter_width = filters_info[ii].first;
            const auto &nb_filters = filters_info[ii].second;

            for (unsigned pp = 0; pp < filter_width - 1; ++pp) { s.push_back(padding); }
            for (unsigned jj = 0; jj < nb_filters; ++jj) {
                auto filter = dynet::expr::parameter(cg, p_filters[ii][jj]);
                auto bias = dynet::expr::parameter(cg, p_biases[ii][jj]);
                auto t = dynet::expr::conv1d_narrow(dynet::expr::concatenate_cols(s), filter);
                t = colwise_add(t, bias);
                t = dynet::expr::kmax_pooling(t, 1);
                tmp.push_back(t);
            }
            for (unsigned p = 0; p < filter_width - 1; ++p) { s.pop_back(); }
        }
        return dynet::expr::rectify(dynet::expr::parameter(cg, p_W) * dynet::expr::concatenate(tmp));
    }


    std::vector<std::vector<dynet::Parameter>> p_filters;
    std::vector<std::vector<dynet::Parameter>> p_biases;
    std::vector<float> zeros;
    std::vector<std::vector<dynet::expr::Expression>> h;
    std::vector<std::pair<unsigned, unsigned>> filters_info;
    dynet::Parameter p_W;

};

#endif //DLNE_EMBEDDING_METHODS_H

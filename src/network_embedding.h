//
// Created by Adoni1203 on 16/7/21.
//

#ifndef DLNE_NETWORK_EMBEDDING_H_H
#define DLNE_NETWORK_EMBEDDING_H_H

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
# include "cnn/expr.h"

#include "graph_data.h"
#include "embedding_methods.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace cnn;

struct DLNEModel {
    LookupParameters *p_u; //lookup table for U nodes
    LookupParameters *p_v; //lookup table for V nodes
    Parameters *W_vv;
    Parameters *W_vc;
    unsigned NODE_SIZE;
    unsigned V_EM_DIM;
    unsigned V_NEG;
    unsigned C_NEG;
    ContentEmbeddingMethod *content_embedding_method;
    std::vector<int> to_be_saved_index;


    explicit DLNEModel(Model &model, unsigned NODE_SIZE, unsigned V_NEG, unsigned C_NEG, unsigned V_EM_DIM,
                       ContentEmbeddingMethod *content_embedding_method)
            : NODE_SIZE(NODE_SIZE), V_EM_DIM(V_EM_DIM), V_NEG(V_NEG),
              C_NEG(C_NEG), content_embedding_method(content_embedding_method) {
        assert(content_embedding_method->C_EM_DIM == V_EM_DIM);
        p_u = model.add_lookup_parameters(NODE_SIZE, {V_EM_DIM});
        p_v = model.add_lookup_parameters(NODE_SIZE, {V_EM_DIM});
        W_vv = model.add_parameters({V_EM_DIM,V_EM_DIM});
        W_vv = model.add_parameters({V_EM_DIM,content_embedding_method->C_EM_DIM});
        init_params();
        std::cout << "Method name: " << content_embedding_method->get_method_name() << std::endl;
        to_be_saved_index.resize(NODE_SIZE);
        std::iota(to_be_saved_index.begin(), to_be_saved_index.end(), 0);
    }

    void init_params() {
        std::vector<float> init(V_EM_DIM);
        for (int j = 0; j < init.size(); j++) {
            init[j] = 0.0;
        }
        for (int i = 0; i < NODE_SIZE; i++) {
            p_v->Initialize(i, init);
        }
        std::uniform_real_distribution<> dis(-0.5, 0.5);
        for (int i = 0; i < NODE_SIZE; i++) {
            for (int j = 0; j < init.size(); j++) {
                init[j] = (float) dis(*cnn::rndeng) / V_EM_DIM;
            }
            p_u->Initialize(i, init);
        }
    }

    void initialize_from_pretrained_vertex_embedding(std::string file_name, GraphData &graph_data) {
        std::cout << "Initializing lookup table from " << file_name << " ..." << std::endl;
        std::ifstream em_in(file_name);
        assert(em_in);
        unsigned em_count, em_size;
        em_in >> em_count >> em_size;
        assert(em_size == V_EM_DIM * 2);
        std::vector<float> eu(em_size / 2);
        std::vector<float> ev(em_size / 2);

        std::string w;
        int initialized_word_count = 0;
        for (int i = 0; i < em_count; i++) {
            em_in >> w;
            unsigned index = graph_data.id_map.get_node_id(w);
            for (int j = 0; j < em_size / 2; j++) {
                em_in >> eu[j];
            }
            for (int j = 0; j < em_size / 2; j++) {
                em_in >> ev[j];
            }
            if (index == -1U) continue;
            initialized_word_count++;
            assert(index < NODE_SIZE);
            p_u->Initialize(index, eu);
            p_v->Initialize(index, ev);
        }
        std::cout << "Initialize " << initialized_word_count << " vertices" << std::endl;
        std::cout << NODE_SIZE - initialized_word_count << " vertices not initialized" << std::endl;
    }

    void set_to_be_saved_index(std::string to_be_saved_index_file_name, GraphData &graph_data) {
        to_be_saved_index.resize(0);
        std::ifstream to_be_saved_index_file_in(to_be_saved_index_file_name);
        assert(to_be_saved_index_file_in);
        std::string line;
        while (getline(to_be_saved_index_file_in, line)) {
            boost::trim(line);
            unsigned id = graph_data.id_map.get_node_id(line);
            assert(id != -1U);
            to_be_saved_index.push_back(id);
        }
    }

    // return Expression of total loss
    cnn::real TrainVVEdge(const Edge edge, GraphData &graph_data) {
        ComputationGraph cg;
        std::vector<Expression> errs;
        Expression i_x_u = lookup(cg, p_u, edge.u);
        auto negative_samples = graph_data.vv_neg_sample(V_NEG + 1, edge);
        Expression i_W_vv = parameter(cg, W_vv);
        for (int v:negative_samples) {
            Expression i_x_v = lookup(cg, p_v, v);
            int relation_type = graph_data.relation_type(edge.u, v);
            if (relation_type == 1) {
//                errs.push_back(log(logistic(dot_product(i_x_u, i_x_v))));
                errs.push_back(log(logistic(i_x_u*i_W_vv*i_x_v)));
            }
            else {
                errs.push_back(log(logistic(-1 * i_x_u*i_W_vv*i_x_v)));
            }
        }

        Expression i_nerr = -1 * sum(errs);
        cnn::real loss = as_scalar(cg.forward());
        cg.backward();
        return loss;
    }

    cnn::real TrainVCEdge(const Edge edge, GraphData &graph_data) {
        ComputationGraph cg;
        std::vector<Expression> errs;
        Expression i_x_u = lookup(cg, p_u, edge.u);
        auto negative_samples = graph_data.vc_neg_sample(V_NEG + 1, edge);
        for (int i = 0; i < negative_samples.size(); i++) {
            int c = negative_samples[i];
            Expression i_x_c = content_embedding_method->get_embedding(graph_data.id_map.id_to_content[c],
                                                                       graph_data.id_map.id_to_tfidf[c], cg);
            if (i == 0) {
                errs.push_back(log(logistic(dot_product(i_x_u, i_x_c))));
            }
            else {
                errs.push_back(log(logistic(-1 * dot_product(i_x_u, i_x_c))));
            }

        }
        Expression i_nerr = -1 * sum(errs);
        cnn::real loss = as_scalar(cg.forward());
        cg.backward();
        return loss;
    }

    void SaveEmbedding(std::string file_name, GraphData &graph_data) {
        std::cout << "Saving to " << file_name << std::endl;
        ComputationGraph cg;
        std::ofstream output_file(file_name);
        output_file << NODE_SIZE << " " << V_EM_DIM * 2 << "\n";
        for (auto node_id:to_be_saved_index) {
            std::string node = graph_data.id_map.id_to_node[node_id];
            output_file << node << " ";
            auto value_u = as_vector(lookup(cg, p_u, node_id).value());
            std::copy(value_u.begin(), value_u.end(), std::ostream_iterator<float>(output_file, " "));
            auto value_v = as_vector(lookup(cg, p_v, node_id).value());
            std::copy(value_v.begin(), value_v.end(), std::ostream_iterator<float>(output_file, " "));
            output_file << "\n";
        }
    }

    float test_tmp(unsigned edge_id, GraphData &graph_data) {
        ComputationGraph cg;
        auto value_u = as_vector(lookup(cg, p_u, graph_data.vv_edgelist[edge_id].u).value());
        return value_u[0];
    }

    std::string get_learner_name() {
        return content_embedding_method->get_method_name();
    }

};

#endif //DLNE_NETWORK_EMBEDDING_H_H

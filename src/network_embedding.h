//
// Created by Adoni1203 on 16/7/21.
//

#ifndef DLNE_NETWORK_EMBEDDING_H_H
#define DLNE_NETWORK_EMBEDDING_H_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include "network_data.h"
#include "embedding_methods.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace dynet;
using namespace std;

struct DLNEModel {
    LookupParameter p_u; //lookup table for U nodes
    LookupParameter p_v; //lookup table for V nodes
    LookupParameter p_c; //lookup table for content nodes
    vector<Parameter> p_relation_matrixes;
    unsigned normal_embedding_node_size;
    unsigned content_embedding_node_size;
    unsigned normal_embedding_dimension;
    unsigned content_embedding_dimension;
    unsigned edge_type_count;
    vector<unsigned> negative_sampling_size;
    ContentEmbeddingMethod *content_embedding_method;
    vector<unsigned> to_be_saved_index;
    vector<float> alpha;
    float beta;

    explicit DLNEModel(Model &lookup_params_model,
                       Model &params_model,
                       unsigned normal_embedding_node_size,
                       unsigned content_embedding_node_size,
                       unsigned embedding_dimension,
                       unsigned edge_type_count,
                       vector<unsigned> negative_sampling_size,
                       ContentEmbeddingMethod *content_embedding_method,
                       vector<float> alpha,
                       float beta,
                       NetworkData &network_data)
        : normal_embedding_node_size(normal_embedding_node_size),
          content_embedding_node_size(content_embedding_node_size),
          normal_embedding_dimension(embedding_dimension / 2),
          content_embedding_dimension(embedding_dimension),
          edge_type_count(edge_type_count),
          negative_sampling_size(negative_sampling_size),
          content_embedding_method(content_embedding_method),
          alpha(alpha),
          beta(beta){
        assert(edge_type_count == negative_sampling_size.size());
        assert(edge_type_count == alpha.size());
        p_u = lookup_params_model.add_lookup_parameters(normal_embedding_node_size, {normal_embedding_dimension});
        p_v = lookup_params_model.add_lookup_parameters(normal_embedding_node_size, {normal_embedding_dimension});
        p_c = lookup_params_model.add_lookup_parameters(content_embedding_node_size, {content_embedding_dimension});
        init_params(network_data);
//    normalize_lookup_table();
        cout << "Content embedding method name: " << content_embedding_method->get_method_name() << endl;
        to_be_saved_index.resize(normal_embedding_node_size);
        iota(to_be_saved_index.begin(), to_be_saved_index.end(), 0);
    }

    void init_params(NetworkData &network_data) {
        vector<float> init(normal_embedding_dimension, 0.0);
        for (unsigned i = 0; i < normal_embedding_node_size; i++) {
            p_v.initialize(i, init);
        }
        uniform_real_distribution<> dis(-0.5, 0.5);
        for (unsigned i = 0; i < normal_embedding_node_size; i++) {
            for (unsigned j = 0; j < init.size(); j++) {
                init[j] = (float) dis(*dynet::rndeng) / normal_embedding_dimension;
            }
            p_u.initialize(i, init);
        }
        vector<float> c_init(content_embedding_dimension, 0.0);
        for (unsigned i = 0; i < content_embedding_node_size; i++) {
            p_c.initialize(i, c_init);
        }
//    for (auto node:network_data.node_list){
//      ComputationGraph cg;
//      auto content_embedding=content_embedding_method->get_embedding(node.content,cg);
//      p_c.initialize(node.embedding_id, as_vector(content_embedding.value()));
//    }
    }

    void set_to_be_saved_index(string to_be_saved_index_file_name, NetworkData &network_data) {
        to_be_saved_index.resize(0);
        ifstream to_be_saved_index_file_in(to_be_saved_index_file_name);
        assert(to_be_saved_index_file_in);
        string line;
        while (getline(to_be_saved_index_file_in, line)) {
            boost::trim(line);
            unsigned node_id = network_data.node_id_map.convert(line);
            to_be_saved_index.push_back(node_id);
        }
    }

    void normalize_vector(vector<float> &value) {
        float min_value = *min_element(value.begin(), value.end());
        float max_value = *max_element(value.begin(), value.end());
        if (min_value == max_value) {
            return;
        }
        for (int i = 0; i < value.size(); i++) {
            value[i] = (value[i] - min_value) / (max_value - min_value);
            assert(value[i] >= 0 && value[i] <= 1);
        }
    }
    void normalize_lookup_table() {
        cout << "Normalizing ... ";
        for (unsigned i = 0; i < normal_embedding_node_size; i++) {
            ComputationGraph cg;
            auto value_u = as_vector(lookup(cg, p_u, i).value());
            normalize_vector(value_u);
            p_u.initialize(i, value_u);
            auto value_v = as_vector(lookup(cg, p_v, i).value());
            normalize_vector(value_v);
            p_v.initialize(i, value_v);
        }
        for (unsigned i = 0; i < content_embedding_node_size; i++) {
            ComputationGraph cg;
            auto value_c = as_vector(lookup(cg, p_c, i).value());
            normalize_vector(value_c);
            p_c.initialize(i, value_c);
        }
        for (unsigned i = 0; i < content_embedding_method->W_EM_SIZE; i++) {
            ComputationGraph cg;
            auto value_w = as_vector(lookup(cg, content_embedding_method->pw, i).value());
            normalize_vector(value_w);
            content_embedding_method->pw.initialize(i, value_w);
        }
        cout << "Done" << endl;
    }
    // return Expression of total loss
    dynet::real Train(const NetEdge edge, NetworkData &network_data) {
        ComputationGraph cg;
        vector<Expression> errs;
        Expression i_x_u = get_node_embedding_u(edge, network_data, cg);
        Expression i_x_v = get_node_embedding_v(edge, network_data, cg);

        Expression score;
        score = simple_score(i_x_u, i_x_v);

        errs.push_back(log(logistic(score)));

        auto neg_edges = network_data.edge_neg_sample(negative_sampling_size[edge.edge_type], edge);
        for (auto neg_edge:neg_edges) {
            Expression neg_i_x_u = get_node_embedding_u(neg_edge, network_data, cg);
            Expression neg_i_x_v = get_node_embedding_v(neg_edge, network_data, cg);

            Expression neg_score;
            neg_score = simple_score(neg_i_x_u, neg_i_x_v);

            errs.push_back(log(logistic(-1 * neg_score)));
        }

        Expression i_nerr = -1 * alpha[edge.edge_type] * sum(errs);
        dynet::real loss = as_scalar(cg.forward(i_nerr));
        cg.backward(i_nerr);
        return loss;
    }

    Expression get_node_embedding_u(const NetEdge &edge, NetworkData &network_data, ComputationGraph &cg) {
        if (network_data.node_list[edge.v_id].with_content) {
            Expression as_u = lookup(cg, p_u, network_data.node_list[edge.u_id].embedding_id);
            Expression as_v = lookup(cg, p_v, network_data.node_list[edge.u_id].embedding_id);
            return concatenate({as_u, as_v});
        } else {
            return lookup(cg, p_u, network_data.node_list[edge.u_id].embedding_id);
        }
    }

    Expression get_node_embedding_v(const NetEdge &edge, NetworkData &network_data, ComputationGraph &cg) {
        if (network_data.node_list[edge.v_id].with_content) {
            uniform_real_distribution<> dis(0.0, 1.0);
            if ((float) dis(*dynet::rndeng) < beta) {
                Expression lookup_embedding = lookup(cg, p_c, network_data.node_list[edge.v_id].embedding_id);
                return lookup_embedding;
            } else {
                Expression content_embedding =
                    content_embedding_method->get_embedding(network_data.node_list[edge.v_id].content, cg);
                return content_embedding;
            }
        } else {
            return lookup(cg, p_v, network_data.node_list[edge.v_id].embedding_id);
        }
    }

    Expression simple_score(Expression &i_x_u, Expression &i_x_v) {
        return dot_product(i_x_u, i_x_v);
    }

    void SaveEmbedding(string embedding_file_name, string relation_file_name, NetworkData &network_data) {
        cout << "Saving embedding to " << embedding_file_name << endl;
        cout << "Saving relation to " << relation_file_name << endl;
//    normalize_lookup_table();

        ofstream embedding_fout(embedding_file_name);
        embedding_fout << to_be_saved_index.size() << " " << normal_embedding_dimension * 2 << "\n";
        for (auto node_id:to_be_saved_index) {
            ComputationGraph cg;
            string node = network_data.node_id_map.convert(node_id);
            embedding_fout << node << " ";
            auto value_u = as_vector(lookup(cg, p_u, node_id).value());
            auto value_v = as_vector(lookup(cg, p_v, node_id).value());
            auto it = value_u.end();
            value_u.insert(it, value_v.begin(), value_v.end());
            normalize_vector(value_u);
            copy(value_u.begin(), value_u.end(), ostream_iterator<float>(embedding_fout, " "));
            embedding_fout << "\n";
        }
        embedding_fout.close();
    }

    string get_learner_name() {
        return content_embedding_method->get_method_name();
    }

};

#endif //DLNE_NETWORK_EMBEDDING_H_H

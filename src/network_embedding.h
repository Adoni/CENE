//
// Created by Adoni1203 on 16/7/21.
//

#ifndef DLNE_NETWORK_EMBEDDING_H_H
#define DLNE_NETWORK_EMBEDDING_H_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
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
    vector<Parameter> p_relation_matrixes;
    unsigned embedding_node_size;
    unsigned embedding_dimension;
    vector<unsigned> negative_sampling_size;
    ContentEmbeddingMethod *content_embedding_method;
    vector<unsigned> to_be_saved_index;
    vector<float> alpha;


    explicit DLNEModel(Model &params_model, unsigned embedding_node_size,
                       unsigned embedding_dimension, unsigned edge_type_count, vector<unsigned> negative_sampling_size,
                       ContentEmbeddingMethod *content_embedding_method, vector<float> alpha)
            : embedding_node_size(embedding_node_size), embedding_dimension(embedding_dimension),
              negative_sampling_size(negative_sampling_size),
              content_embedding_method(content_embedding_method), alpha(alpha) {
        assert(edge_type_count == negative_sampling_size.size());
        assert(edge_type_count == alpha.size());
        p_u = params_model.add_lookup_parameters(embedding_node_size, {embedding_dimension});
        p_v = params_model.add_lookup_parameters(embedding_node_size, {embedding_dimension});
        p_relation_matrixes.resize(0);
        for (int i = 0; i < edge_type_count; i++) {
            p_relation_matrixes.push_back(params_model.add_parameters({embedding_dimension}));
        }
        init_params();
        cout << "Content embedding method name: " << content_embedding_method->get_method_name() << endl;
        to_be_saved_index.resize(embedding_node_size);
        iota(to_be_saved_index.begin(), to_be_saved_index.end(), 0);
    }

    void init_params() {
        vector<float> init(embedding_dimension, 0.0);
        for (unsigned i = 0; i < embedding_node_size; i++) {
            p_v.initialize(i, init);
        }
        uniform_real_distribution<> dis(-0.5, 0.5);
        for (unsigned i = 0; i < embedding_node_size; i++) {
            for (unsigned j = 0; j < init.size(); j++) {
                init[j] = (float) dis(*dynet::rndeng) / embedding_dimension;
            }
            p_u.initialize(i, init);
        }
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

    // return Expression of total loss
    dynet::real Train(const Edge edge, NetworkData &network_data) {
        ComputationGraph cg;
        vector<Expression> errs;
        Expression i_x_u;
        if (network_data.node_list[edge.u_id].with_content) {
            i_x_u = content_embedding_method->get_embedding(network_data.node_list[edge.u_id].content, cg);
        } else {
            i_x_u = lookup(cg, p_u, network_data.node_list[edge.u_id].embedding_id);
        }
        auto negative_samples = network_data.vv_neg_sample(negative_sampling_size[edge.edge_type] + 1, edge);
        // Expression i_W_vv = parameter(cg, W_vv);
        for (int i = 0; i < negative_samples.size(); i++) {
            int v_id = negative_samples[i];
            Expression i_x_v;
            if (network_data.node_list[edge.v_id].with_content) {
                i_x_v = content_embedding_method->get_embedding(network_data.node_list[v_id].content, cg);
            } else {
                i_x_v = lookup(cg, p_v, network_data.node_list[v_id].embedding_id);
            }
            Expression score = bilinear_score(i_x_u, i_x_v, cg, edge.edge_type);
            if (i == 0) {
                errs.push_back(log(logistic(score)));
            } else {
                errs.push_back(log(logistic(-1 * score)));
            }
        }

        Expression i_nerr = -1 * alpha[edge.edge_type] * sum(errs);
        dynet::real loss = as_scalar(cg.forward(i_nerr));
        cg.backward(i_nerr);
        return loss;
    }

    Expression simple_score(Expression &i_x_u, Expression &i_x_v) {
        return dot_product(i_x_u, i_x_v);
    }

    Expression bilinear_score(Expression &i_x_u, Expression &i_x_v, ComputationGraph &cg, unsigned edge_type) {
        assert(edge_type < p_relation_matrixes.size());
        Expression W = parameter(cg, p_relation_matrixes[edge_type]);
        return dot_product(softmax(i_x_u), softmax(cmult(W, i_x_v)));
    }

    void SaveEmbedding(string file_name, NetworkData &network_data) {
        cout << "Saving to " << file_name << endl;

        ofstream output_file(file_name);
        output_file << to_be_saved_index.size() << " " << embedding_dimension * 2 << "\n";
        for (auto node_id:to_be_saved_index) {
            ComputationGraph cg;
            string node = network_data.node_id_map.convert(node_id);
            output_file << node << " ";
            auto value_u = as_vector(lookup(cg, p_u, node_id).value());
            copy(value_u.begin(), value_u.end(), ostream_iterator<float>(output_file, " "));
            auto value_v = as_vector(lookup(cg, p_v, node_id).value());
            copy(value_v.begin(), value_v.end(), ostream_iterator<float>(output_file, " "));
            output_file << "\n";
        }
    }

    string get_learner_name() {
        return content_embedding_method->get_method_name();
    }

};

#endif //DLNE_NETWORK_EMBEDDING_H_H

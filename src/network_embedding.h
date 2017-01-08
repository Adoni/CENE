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
            vector<float> init(embedding_dimension, 1.0);
            ParameterInitFromVector matrix_init(init);
            p_relation_matrixes.push_back(params_model.add_parameters({embedding_dimension}, matrix_init));
        }
        init_params();
        cout << "Content embedding method name: " << content_embedding_method->get_method_name() << endl;
        to_be_saved_index.resize(embedding_node_size);
        iota(to_be_saved_index.begin(), to_be_saved_index.end(), 0);
    }

    void init_params() {
        vector<float> init(embedding_dimension, 0.0);

        uniform_real_distribution<> dis(-0.5, 0.5);
        for (unsigned i = 0; i < embedding_node_size; i++) {
//            for (unsigned j = 0; j < init.size(); j++) {
//                init[j] = (float) dis(*dynet::rndeng) / embedding_dimension;
//            }
            p_v.initialize(i, init);
        }

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
        Expression i_x_u = get_node_embedding(edge.u_id, network_data, cg, true);
        Expression i_x_v = get_node_embedding(edge.v_id, network_data, cg, false);

//        Expression score = simple_score(i_x_u, i_x_v);
        Expression score = bilinear_score(i_x_u, i_x_v, cg, edge.edge_type);
        errs.push_back(log(logistic(score)));

        auto v_id_negative_samples = network_data.v_id_neg_sample(negative_sampling_size[edge.edge_type], edge);
        for (auto neg_v_id:v_id_negative_samples) {
            Expression neg_i_x_v = get_node_embedding(neg_v_id, network_data, cg, true);
//            Expression neg_score = simple_score(i_x_u, neg_i_x_v);
            Expression neg_score = bilinear_score(i_x_u, neg_i_x_v, cg, edge.edge_type);
            errs.push_back(log(logistic(-1 * neg_score)));
        }

        auto edge_type_negative_samples = network_data.edge_type_neg_sample(negative_sampling_size[edge.edge_type],
                                                                            edge);
        for (auto neg_edge_type:edge_type_negative_samples) {
            Expression neg_score = bilinear_score(i_x_u, i_x_v, cg, neg_edge_type);
            errs.push_back(log(logistic(-1 * neg_score)));
        }

        Expression i_nerr = -1 * alpha[edge.edge_type] * sum(errs);
        dynet::real loss = as_scalar(cg.forward(i_nerr));
        cg.backward(i_nerr);
        return loss;
    }

    Expression get_node_embedding(int node_id, NetworkData &network_data, ComputationGraph &cg, bool use_p_u){
        if (network_data.node_list[node_id].with_content) {
            return content_embedding_method->get_embedding(network_data.node_list[node_id].content, cg);
        } else {
            if (use_p_u) {
                return lookup(cg, p_u, network_data.node_list[node_id].embedding_id);
            }else{
                return lookup(cg, p_v, network_data.node_list[node_id].embedding_id);
            }
        }
    }
    Expression simple_score(Expression &i_x_u, Expression &i_x_v) {
        return dot_product(i_x_u, i_x_v);
    }

    Expression bilinear_score(Expression &i_x_u, Expression &i_x_v, ComputationGraph &cg, int edge_type) {
        assert(edge_type < p_relation_matrixes.size());
        Expression W = parameter(cg, p_relation_matrixes[edge_type]);
        return dot_product(i_x_u, cmult(W, i_x_v));
    }

    void SaveEmbedding(string file_name, NetworkData &network_data) {
        cout << "Saving to " << file_name << endl;
        for (int i = 0; i < p_relation_matrixes.size(); i++) {
            cout << "W" << i << endl;
            ComputationGraph cg;
            auto w = as_vector(parameter(cg, p_relation_matrixes[i]).value());
            for (auto wi:w) {
                cout << wi << " ";
            }
            cout << endl;
        }

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

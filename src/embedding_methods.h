//
// Created by Adoni1203 on 16/7/20.
//

#ifndef DLNE_EMBEDDING_METHODS_H
#define DLNE_EMBEDDING_METHODS_H

#include "graph_data.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "cnn/gru.h"
#include "cnn/dict.h"
#include <vector>
#include <string>

using namespace cnn;

class ContentEmbeddingMethod {
public:
    virtual ~ContentEmbeddingMethod() { }

    virtual Expression get_embedding(const CONTENT_TYPE &content, const TFIDF_TYPE &tfidf, ComputationGraph &cg) = 0;

    std::string get_method_name() {
        return method_name;
    }

    void initial_look_up_table_from_file(std::string file_name, Dict &d) {
        std::cout << "Initializing lookup table from " << file_name << " ..." << std::endl;
        std::string UNK = "UNKNOWN_WORD";
        std::ifstream em_in(file_name);
        assert(em_in);
        unsigned em_count, em_size;
        int unknow_id = d.Convert(UNK);
        em_in >> em_count >> em_size;
        assert(em_size == W_EM_DIM);
        std::vector<float> e(em_size);
        std::string w;
        int initialized_word_count = 0;
        for (int i = 0; i < em_count; i++) {
            em_in >> w;
            for (int j = 0; j < em_size; j++) {
                em_in >> e[j];
            }
            int index = d.Convert(w);
            if (index == unknow_id) continue;
            initialized_word_count++;
            assert(index < d.size() && index >= 0);
            p->Initialize(index, e);
        }
        std::cout << "Initialize " << initialized_word_count << " words" << std::endl;
        std::cout << d.size() - initialized_word_count << " words not initialized" << std::endl;
    }

    void initial_look_up_table(unsigned lookup_table_size) {
        std::vector<float> init(W_EM_DIM);
        std::uniform_real_distribution<> dis(-0.5, 0.5);
        for (int i = 0; i < lookup_table_size; i++) {
            for (int j = 0; j < init.size(); j++) {
                init[j] = (float) dis(*cnn::rndeng) / W_EM_DIM;
            }
            p->Initialize(i, init);
        }
    }

    LookupParameters *p;
    unsigned W_EM_DIM;
    unsigned C_EM_DIM;
    std::string method_name;
};

class WordAvg_CE : public ContentEmbeddingMethod {
public:
    explicit WordAvg_CE(
            Model &model,
            unsigned word_embedding_size,
            unsigned content_embedding_size,
            Dict &d) {
        assert(word_embedding_size == content_embedding_size);
        this->W_EM_DIM = word_embedding_size;
        this->C_EM_DIM = content_embedding_size;
        this->method_name = "WordAvg";
        p = model.add_lookup_parameters(d.size(), {W_EM_DIM});
        initial_look_up_table(d.size());
    }

    ~WordAvg_CE() { }

    Expression get_embedding(const CONTENT_TYPE &content, const TFIDF_TYPE &tfidf, ComputationGraph &cg) {
        std::vector<Expression> all_word_embedding;
        assert(content.size() == tfidf.size());
        for (int i = 0; i < content.size(); i++) {
            std::vector<Expression> sentence_expression;
            for (int j = 0; j < content[i].size(); j++) {
                sentence_expression.push_back(const_lookup(cg, p, content[i][j]));
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
            Model &model,
            unsigned word_embedding_size,
            unsigned content_embedding_size,
            Dict &d) {
        this->W_EM_DIM = word_embedding_size;
        this->C_EM_DIM = content_embedding_size;
        this->method_name = "GRU";
        builder = GRUBuilder(1, W_EM_DIM, C_EM_DIM, &model);
        p = model.add_lookup_parameters(d.size(), {W_EM_DIM});
        initial_look_up_table(d.size());
    }

    ~GRU_CE() { }

    Expression get_embedding(const CONTENT_TYPE &content, const TFIDF_TYPE &tfidf, ComputationGraph &cg) {
        std::vector<Expression> all_hidden;

        for (auto c:content) {
            builder.new_graph(cg);
            builder.start_new_sequence();
            for (auto w:c) {
                Expression i_x_t = lookup(cg, p, w);
                builder.add_input(i_x_t);
            }
            all_hidden.push_back(builder.back());
        }
        return average(all_hidden);
    }
};

class CNN_CE : public ContentEmbeddingMethod {
public:
    explicit CNN_CE(cnn::Model &model, unsigned input_dim, unsigned output_dim,
           const std::vector<std::pair<unsigned, unsigned>> &info) :
            zeros(input_dim, 0.), filters_info(info) {
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
                p_filters[i][j] = model.add_parameters({input_dim, filter_width});
                p_biases[i][j] = model.add_parameters({input_dim});
                combined_dim += input_dim;
            }
        }
        p_W = model.add_parameters({output_dim, combined_dim});
    }


    cnn::expr::Expression get_embedding(cnn::ComputationGraph *cg, std::vector<cnn::expr::Expression> &c) {
        unsigned len = c.size();
        auto padding = cnn::expr::zeroes(*cg, {(unsigned int)zeros.size()});
        std::vector<cnn::expr::Expression> s(c);
        std::vector<cnn::expr::Expression> tmp;
        for (unsigned ii = 0; ii < filters_info.size(); ++ii) {
            const auto &filter_width = filters_info[ii].first;
            const auto &nb_filters = filters_info[ii].second;

            for (unsigned p = 0; p < filter_width - 1; ++p) { s.push_back(padding); }
            for (unsigned jj = 0; jj < nb_filters; ++jj) {
                auto filter = cnn::expr::parameter(*cg, p_filters[ii][jj]);
                auto bias = cnn::expr::parameter(*cg, p_biases[ii][jj]);
                auto t = cnn::expr::conv1d_narrow(cnn::expr::concatenate_cols(s), filter);
                t = colwise_add(t, bias);
                t = cnn::expr::rectify(cnn::expr::kmax_pooling(t, 1));
                tmp.push_back(t);
            }
            for (unsigned p = 0; p < filter_width - 1; ++p) { s.pop_back(); }
        }
        return cnn::expr::parameter(*cg, p_W) * cnn::expr::concatenate(tmp);
    }

    std::vector<std::vector<cnn::Parameters*>> p_filters;
    std::vector<std::vector<cnn::Parameters*>> p_biases;
    std::vector<float> zeros;
    std::vector<std::vector<cnn::expr::Expression>> h;
    std::vector<std::pair<unsigned, unsigned>> filters_info;
    cnn::Parameters* p_W;

};

#endif //DLNE_EMBEDDING_METHODS_H

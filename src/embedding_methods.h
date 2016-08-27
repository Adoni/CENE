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
#include "cnn/training.h"
#include <vector>
#include <string>

using namespace cnn;

class ContentEmbeddingMethod {
public:
    virtual ~ContentEmbeddingMethod() { }

    virtual Expression get_embedding(const CONTENT_TYPE &content, const TFIDF_TYPE &tfidf, ComputationGraph &cg) = 0;

    virtual void pretraining(GraphData &graph_data,Model &model)=0;

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
    bool use_const_lookup;
};

class WordAvg_CE : public ContentEmbeddingMethod {
public:
    explicit WordAvg_CE(
            Model &model,
            unsigned word_embedding_size,
            unsigned content_embedding_size,
            bool use_const_lookup,
            Dict &d) {
        assert(word_embedding_size == content_embedding_size);
        this->W_EM_DIM = word_embedding_size;
        this->C_EM_DIM = content_embedding_size;
        this->method_name = "WordAvg";
        this->use_const_lookup = use_const_lookup;
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
                if(use_const_lookup){
                    sentence_expression.push_back(const_lookup(cg, p, content[i][j]));
                }
                else{
                    sentence_expression.push_back(lookup(cg, p, content[i][j]));
                }
            }
            all_word_embedding.push_back(average(sentence_expression));
        }
        return average(all_word_embedding);
    }
    void pretraining(GraphData &graph_data,Model &model){}
};

class GRU_CE : public ContentEmbeddingMethod {
public:
    GRUBuilder builder;
    Parameters* p_R;
    Parameters* p_bias;

    explicit GRU_CE(
            Model &model,
            unsigned word_embedding_size,
            unsigned content_embedding_size,
            bool use_const_lookup,
            GraphData &graph_data,
            Dict &d) {
        this->W_EM_DIM = word_embedding_size;
        this->C_EM_DIM = content_embedding_size;
        this->method_name = "GRU";
        this->use_const_lookup=use_const_lookup;

        builder = GRUBuilder(1, W_EM_DIM, C_EM_DIM, &model);
        p = model.add_lookup_parameters(d.size(), {W_EM_DIM});
        p_R = model.add_parameters({d.size(), content_embedding_size});
        p_bias = model.add_parameters({d.size()});
        initial_look_up_table(d.size());
        pretraining(graph_data, model);
    }

    ~GRU_CE() { }

    Expression get_embedding(const CONTENT_TYPE &content, const TFIDF_TYPE &tfidf, ComputationGraph &cg) {
        assert(content.size()==1);
        builder.new_graph(cg);
        builder.start_new_sequence();
        std::vector<Expression> all_hidden;
        for (auto w:content[0]) {
            if(use_const_lookup){
                all_hidden.push_back(builder.add_input(const_lookup(cg, p, w)));
            }
            else{
                all_hidden.push_back(builder.add_input(lookup(cg, p, w)));
            }
        }
        return average(all_hidden);
    }

    void pretraining(GraphData &graph_data, Model &model){
        std::cout<<"Pretraining"<<std::endl;
        Trainer* sgd = nullptr;
        sgd = new SimpleSGDTrainer(&model);
        int counter=0;
        int total=graph_data.id_map.id_to_content.size();
        float loss=0.0;
        for(auto& content:graph_data.id_map.id_to_content){
            std::cout<<"Pretraining: "<<counter<<"/"<<total<<std::endl;
            for(auto& sent:content){
                ComputationGraph cg;
                sentence_pretraining(sent,cg);
                loss += as_scalar(cg.forward());
                cg.backward();
                sgd->update();
            }
            counter+=1;
        }
    }

    Expression sentence_pretraining(const SENT_TYPE  &sent, ComputationGraph& cg){
        const unsigned slen = sent.size();
        builder.new_graph(cg);  // reset RNN builder for new graph
        builder.start_new_sequence();
        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias
        std::vector<Expression> errs;
        for (unsigned t = 0; t < slen-1; ++t) {
            Expression i_x_t = lookup(cg, p, sent[t]);
            Expression i_y_t = builder.add_input(i_x_t);
            Expression i_r_t =  i_bias + i_R * i_y_t;
            Expression i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
            errs.push_back(i_err);
        }
        Expression i_nerr = sum(errs);
        return i_nerr;
    }
};

class CNN_CE : public ContentEmbeddingMethod {
public:
    explicit CNN_CE(cnn::Model &model, unsigned word_embedding_size, unsigned content_embedding_size,
           const std::vector<std::pair<unsigned, unsigned>> &info, Dict &d) :
            zeros(word_embedding_size, 0.), filters_info(info) {
        this->W_EM_DIM = word_embedding_size;
        this->C_EM_DIM = content_embedding_size;
        this->method_name = "CNN";
        p = model.add_lookup_parameters(d.size(), {W_EM_DIM});
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
                p_filters[i][j] = model.add_parameters({W_EM_DIM, filter_width});
                p_biases[i][j] = model.add_parameters({W_EM_DIM});
                combined_dim += W_EM_DIM;
            }
        }
        p_W = model.add_parameters({C_EM_DIM, combined_dim});
    }


    cnn::expr::Expression get_embedding(const CONTENT_TYPE &content, const TFIDF_TYPE &tfidf, ComputationGraph &cg) {
        unsigned len = content.size();
        auto padding = cnn::expr::zeroes(cg, {(unsigned int)zeros.size()});
        std::vector<cnn::expr::Expression> s;
        for(auto c:content){
            for(auto w:c){
                s.push_back(lookup(cg, p, w));
            }
        }
        std::vector<cnn::expr::Expression> tmp;
        for (unsigned ii = 0; ii < filters_info.size(); ++ii) {
            const auto &filter_width = filters_info[ii].first;
            const auto &nb_filters = filters_info[ii].second;

            for (unsigned p = 0; p < filter_width - 1; ++p) { s.push_back(padding); }
            for (unsigned jj = 0; jj < nb_filters; ++jj) {
                auto filter = cnn::expr::parameter(cg, p_filters[ii][jj]);
                auto bias = cnn::expr::parameter(cg, p_biases[ii][jj]);
                auto t = cnn::expr::conv1d_narrow(cnn::expr::concatenate_cols(s), filter);
                t = colwise_add(t, bias);
                t = cnn::expr::rectify(cnn::expr::kmax_pooling(t, 1));
                tmp.push_back(t);
            }
            for (unsigned p = 0; p < filter_width - 1; ++p) { s.pop_back(); }
        }
        return cnn::expr::parameter(cg, p_W) * cnn::expr::concatenate(tmp);
    }
    void pretraining(GraphData &graph_data,Model &model){}


    std::vector<std::vector<cnn::Parameters*>> p_filters;
    std::vector<std::vector<cnn::Parameters*>> p_biases;
    std::vector<float> zeros;
    std::vector<std::vector<cnn::expr::Expression>> h;
    std::vector<std::pair<unsigned, unsigned>> filters_info;
    cnn::Parameters* p_W;

};

#endif //DLNE_EMBEDDING_METHODS_H

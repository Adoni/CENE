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

    virtual Expression get_embedding(const SENT_TYPE &content, ComputationGraph &cg) = 0;

    std::string get_method_name() {
        return method_name;
    }
    void initial_look_up_table(std::string file_name, Dict &d) {
        std::cout << "Initializing lookup table from " << file_name << " ..." <<std::endl;
        std::string UNK="UNKNOWN_WORD";
        std::ifstream em_in(file_name);
        assert(em_in);
        unsigned em_count, em_size;
        unsigned unknow_id = d.Convert(UNK);
        em_in >> em_count >> em_size;
        assert(em_size==INPUT_SIZE);
        std::vector<float> e(em_size);
        std::string w;
        int initialized_word_count=0;
        for (int i = 0; i < em_count; i++) {
            em_in >> w;
            for (int j = 0; j < em_size; j++) {
                em_in >> e[j];
            }
            unsigned index = d.Convert(w);
            if (index == unknow_id) continue;
            initialized_word_count++;
            assert(index<d.size() && index>=0);
            p->Initialize(index, e);
        }
        std::cout << "Initialize " << initialized_word_count << " words" << std::endl;
        std::cout << d.size() - initialized_word_count << " words not initialized" << std::endl;
    }
    LookupParameters *p;
    unsigned INPUT_SIZE;
    unsigned OUTPUT_SIZE;
    std::string method_name;
};

class WordAvg : public ContentEmbeddingMethod {
public:
    explicit WordAvg(
            Model& model,
            unsigned word_embedding_size,
            unsigned content_embedding_size,
            std::string word_embedding_file,
            Dict &d){
        assert(word_embedding_size==content_embedding_size);
        this->INPUT_SIZE = word_embedding_size;
        this->OUTPUT_SIZE = content_embedding_size;
        this->method_name = "WordAvg";
        p = model.add_lookup_parameters(d.size(), {INPUT_SIZE});
//        initial_look_up_table(word_embedding_file, d);
    }

    ~WordAvg() {}

    Expression get_embedding(const SENT_TYPE &content, ComputationGraph &cg){
        std::vector<Expression> all_word_embedding;
        for (auto w:content){
            all_word_embedding.push_back(lookup(cg, p, w));
        }
        return average(all_word_embedding);
    }
};
#endif //DLNE_EMBEDDING_METHODS_H

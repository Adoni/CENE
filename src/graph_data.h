//
// Created by Adoni1203 on 16/7/19.
// VV: vertex to vertex
// VC: vertex to content
//

#ifndef WEIBONETEMBEDDING_GRAPH_DATA_H
#define WEIBONETEMBEDDING_GRAPH_DATA_H

#include "cnn/cnn.h"
#include "cnn/dict.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <unordered_set>
#include <boost/algorithm/string.hpp>

struct Edge{
    unsigned u;
    unsigned v;
};
typedef std::vector<int> SENT_TYPE; //保存句子信息
typedef std::vector<SENT_TYPE> CONTENT_TYPE; //保存content信息
typedef std::vector<std::vector<float>> TFIDF_TYPE;

struct ID_MAP{
    std::vector<std::string> id_to_node;
    std::unordered_map<std::string, unsigned> node_to_int;
    std::vector<CONTENT_TYPE> id_to_content;
    std::vector<TFIDF_TYPE> id_to_tfidf;
    bool frozen;
    ID_MAP(): frozen(false) {};
    unsigned get_node_id(std::string node){
        auto i=node_to_int.find(node);
        if (i !=node_to_int.end()){
            return i->second;
        }
        else{
            if(!frozen){
                unsigned nid=node_to_int.size();
                node_to_int[node]=nid;
                id_to_node.push_back(node);
                assert(id_to_node[nid]==node);
                return node_to_int.size()-1;
            }
            else{
                return -1U;
            }

        }
    }
    unsigned get_content_id(CONTENT_TYPE content){
        id_to_content.push_back(content);
        return id_to_content.size()-1;
    }
    void Freeze() { frozen = true; }
};


struct GraphData {
    ID_MAP id_map;
    unsigned node_count; // 节点数量
    std::vector<std::unordered_set<unsigned>> vv_utov_graph; //保存以u为中心点的vv邻接表
    std::vector<std::unordered_set<unsigned>> vv_vtou_graph; //保存以v为中心点的vc邻接表,作用是构建负采样表
    std::vector<std::unordered_set<unsigned>> vc_graph;
    unsigned long vv_table_size; //vv负采样表的大小
    std::vector<Edge> vv_edgelist; //保存所有vv边
    std::vector<Edge> vc_edgelist; //保存所有vc边
    std::vector<unsigned> vv_unitable;


    explicit GraphData(std::string graph_file_name,
                       std::string content_file_name,
                       bool strictly_content_required,
                       cnn::Dict &d) {
        std::cout << "Graph file: " << graph_file_name << std::endl;
        std::cout << "Content file: " << content_file_name << std::endl;

        get_node_count(graph_file_name, content_file_name, strictly_content_required);
        std::cout<<"Node count: "<<node_count<<std::endl;
        read_graph_from_file(graph_file_name);
        read_content_from_file(content_file_name, d);
        vv_table_size=1e8;
        InitUnigramTable();
        std::string UNK = "UNKNOWN_WORD";
        d.Freeze();
        d.SetUnk(UNK);
    }

    void get_node_count(std::string graph_file_name, std::string content_file_name, bool strictly_content_required){
        std::cout<<"Strict: "<<strictly_content_required<<std::endl;
        std::string line;

        std::ifstream content_in(content_file_name);
        assert(content_in);
        while (getline(content_in, line)) {
            boost::trim(line);
            std::string::size_type split_pos = line.find(' ');
            std::string sub = line.substr(0, split_pos - 0);
            id_map.get_node_id(sub);
        }


        if(strictly_content_required){
            id_map.Freeze();
        }

        std::ifstream graph_in(graph_file_name);
        assert(graph_in);
        while (getline(graph_in, line)) {
            std::istringstream buffer(line);
            std::vector<std::string> nodes((std::istream_iterator<std::string>(buffer)),
                                      std::istream_iterator<std::string>());

            for (auto node:nodes) {
                id_map.get_node_id(node);
            }
        }
        node_count=id_map.node_to_int.size();
        id_map.Freeze();
    }

    void read_graph_from_file(std::string graph_file_name){
        std::cout<<"Load graph"<<std::endl;
        std::string line;

        vv_utov_graph.resize(node_count);
        vv_vtou_graph.resize(node_count);

        std::ifstream graph_in(graph_file_name);
        assert(graph_in);
        while (getline(graph_in, line)) {
            boost::trim(line);
            std::istringstream buffer(line);
            std::vector<std::string> nodes((std::istream_iterator<std::string>(buffer)),
                               std::istream_iterator<std::string>());

            std::vector<unsigned> node_id;
            for (auto node:nodes){
                unsigned id=id_map.get_node_id(node);
                if(id==-1U){
                    continue;
                }
                node_id.push_back(id);
            }
            for (int i=1;i<node_id.size();i++) {
                vv_utov_graph[node_id[0]].insert(node_id[i]);
                vv_edgelist.push_back(Edge{node_id[0],node_id[i]});
            }
            for (int i=1;i<node_id.size();i++) {
                vv_vtou_graph[node_id[i]].insert(node_id[0]);
            }
        }

    }

    void read_content_from_file(std::string content_file_name, cnn::Dict &d){
        std::cout<<"Load content"<<std::endl;
        std::string line;
        std::ifstream content_in(content_file_name);
        assert(content_in);
        vc_graph.resize(node_count);
        while (getline(content_in, line)) {
            std::string::size_type split_pos = line.find(' ');
            std::string sub = line.substr(0, split_pos - 0);
            unsigned node_id = id_map.get_node_id(sub);
            if (node_id >= node_count) {
                std::cerr << "Error!" << std::endl;
                std::cout << node_id << std::endl;
                exit(0);
            }
            std::string::size_type start_pos=split_pos + 1;
            CONTENT_TYPE content;
            TFIDF_TYPE tfidf;
            while(1){
                std::string::size_type end_pos=line.find("||||", start_pos + 1);
                if(end_pos==std::string::npos){
                    break;
                }
                sub = line.substr(start_pos, end_pos-start_pos);
                content.push_back(ReadSentence(sub, &d));
                tfidf.push_back(std::vector<float>(content.back().size(),  1.0/content.back().size()));
                start_pos=end_pos+4;
            }

            unsigned content_id=id_map.get_content_id(content);
            id_map.id_to_tfidf.push_back(tfidf);
            vc_graph[node_id].insert(content_id);
            vc_edgelist.push_back(Edge{node_id,content_id});
        }
    }

    void read_tfidf_from_file(std::string tfidf_file_name){
        std::cout<<"Load tf-idf"<<std::endl;
        std::string line;
        std::ifstream content_in(tfidf_file_name);
        assert(content_in);
        id_map.id_to_tfidf.clear();
        assert(id_map.id_to_tfidf.size()==0);
        while (getline(content_in, line)) {
            std::string::size_type split_pos = line.find(' ');
            std::string sub = line.substr(0, split_pos - 0);
            unsigned node_id = id_map.get_node_id(sub);
            if (node_id >= node_count) {
                std::cerr << "Error!" << std::endl;
                std::cout << node_id << std::endl;
                exit(0);
            }
            std::string::size_type start_pos=split_pos + 1;
            TFIDF_TYPE tfidf;
            while(1){
                std::string::size_type end_pos=line.find("||||", start_pos + 1);
                if(end_pos==std::string::npos){
                    break;
                }
                sub = line.substr(start_pos, end_pos-start_pos);
                std::istringstream buffer(sub);
                tfidf.push_back(std::vector<float>((std::istream_iterator<float>(buffer)),
                                                         std::istream_iterator<float>()));
                start_pos=end_pos+4;
            }
            id_map.id_to_tfidf.push_back(tfidf);
        }
    }

    void InitUnigramTable() {
        vv_unitable.resize(vv_table_size);
        long long normalizer = 0;
        double d1, power = 0.75;
        for (int a = 0; a < node_count; a++) normalizer += pow(vv_vtou_graph[a].size(), power);
        std::cout<<"normalizer: "<<normalizer<<std::endl;
        unsigned i = 0;
        d1 = std::pow(vv_vtou_graph[i].size(), power) / (double) normalizer;
        for (int a = 0; a < vv_table_size; a++) {
            vv_unitable[a] = i;
            if (a / (double) vv_table_size > d1) {
                i++;
                d1 += pow(vv_vtou_graph[i].size(), power) / (double) normalizer;
            }
            if (i >= node_count) i = node_count - 1;
        }
    }

    unsigned relation_type(unsigned u, unsigned v) {
        if (vv_utov_graph[u].find(v) != vv_utov_graph[u].end()) {
            return 1;
        }
        return 0;
    }

    std::vector<unsigned> vv_neg_sample(unsigned sample_size, Edge edge) {
        std::vector<unsigned> vs(sample_size);
        vs[0] = edge.v;
        unsigned i = 1;
        std::random_device rd;
        std::uniform_int_distribution<> dis(0, vv_table_size - 1);
        while (i < sample_size) {
            unsigned nv = vv_unitable[dis(*cnn::rndeng)];
            assert(nv<node_count);
            if (nv == edge.u || nv == edge.v || relation_type(edge.u, nv) == 1) continue;
            vs[i] = nv;
            i++;
        }
        return vs;
    }

    std::vector<unsigned> vc_neg_sample(unsigned sample_size, Edge edge) {
        std::vector<unsigned> cs(sample_size);
        cs[0] = edge.v;
        unsigned i = 1;
        std::random_device rd;
        std::uniform_int_distribution<> dis(0, vc_edgelist.size()-1);
        while (i < sample_size) {
            unsigned edge_id = dis(*cnn::rndeng);
            if (vc_edgelist[edge_id].u == edge.u) continue;
            cs[i] = vc_edgelist[edge_id].v;
            i++;
        }
        return cs;
    }
};

#endif //WEIBONETEMBEDDING_GRAPH_DATA_H

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

struct Edge{
    int u;
    int v;
};
typedef std::vector<int> SENT_TYPE; //保存句子信息

struct ID_MAP{
    std::vector<std::string> id_to_node;
    std::unordered_map<std::string, int> node_to_int;
    std::vector<SENT_TYPE> id_to_content;
    bool frozen;
    ID_MAP(): frozen(false) {};
    int get_node_id(std::string node){
        auto i=node_to_int.find(node);
        if (i !=node_to_int.end()){
            return i->second;
        }
        else{
            if(!frozen){
                node_to_int[node]=node_to_int.size();
                return node_to_int.size()-1;
            }
            else{
                return -1;
            }

        }
    }
    int get_content_id(SENT_TYPE content){
        id_to_content.push_back(content);
        return id_to_content.size()-1;
    }
    void Freeze() { frozen = true; }
};


struct GraphData {
    ID_MAP id_map;
    unsigned node_count; // 节点数量
    std::vector<std::unordered_set<int>> vv_utov_graph; //保存以u为中心点的vv邻接表
    std::vector<std::unordered_set<int>> vv_vtou_graph; //保存以v为中心点的vc邻接表,作用是构建负采样表
    unsigned long vv_table_size; //vv负采样表的大小
    std::vector<Edge> vv_edgelist; //保存所有vv边
    std::vector<Edge> vc_edgelist; //保存所有vc边
    std::vector<unsigned> vv_unitable;


    explicit GraphData(std::string graph_file_name,
                       std::string content_file_name,
                       cnn::Dict &d) {
        std::cout << "Graph file: " << graph_file_name << std::endl;
        std::cout << "Content file: " << content_file_name << std::endl;

        get_node_count(graph_file_name, content_file_name);
        std::cout<<"Node count: "<<node_count<<std::endl;
        read_graph_from_file(graph_file_name);
        read_content_from_file(content_file_name, d);
        vv_table_size=1e8;
        InitUnigramTable();
        std::string UNK = "UNKNOWN_WORD";
        d.Freeze();
        d.SetUnk(UNK);
    }

    void get_node_count(std::string graph_file_name, std::string content_file_name){
        std::string line;

        std::ifstream content_in(content_file_name);
        assert(content_in);
        while (getline(content_in, line)) {
            unsigned split_point = line.find(' ');
            std::string sub = line.substr(0, split_point - 0);
            id_map.get_node_id(sub);
        }

        id_map.Freeze();
        node_count=id_map.node_to_int.size();

        std::ifstream graph_in(graph_file_name);
        assert(graph_in);
        while (getline(graph_in, line)) {
            std::istringstream buffer(line);
            std::vector<std::string> nodes((std::istream_iterator<std::string>(buffer)),
                                      std::istream_iterator<std::string>());

            for (int i=0;i<nodes.size();i++){
                id_map.get_node_id(nodes[i]);
            }
        }
        node_count=id_map.node_to_int.size();
    }

    void read_graph_from_file(std::string graph_file_name){
        std::cout<<"Load graph"<<std::endl;
        std::string line;

        vv_utov_graph.resize(node_count);
        vv_vtou_graph.resize(node_count);

        std::ifstream graph_in(graph_file_name);
        assert(graph_in);
        while (getline(graph_in, line)) {
            std::istringstream buffer(line);
            std::vector<std::string> nodes((std::istream_iterator<std::string>(buffer)),
                               std::istream_iterator<std::string>());

            std::vector<int> node_id;
            for (auto node:nodes){
                int id=id_map.get_node_id(node);
                if(id==-1){
                    continue;
                }
                assert(id<node_count);
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
        while (getline(content_in, line)) {
            int split_point = line.find(' ');
            std::string sub = line.substr(0, split_point - 0);
            int node_id = id_map.get_node_id(sub);
            if (node_id >= node_count) {
                std::cerr << "Error!" << std::endl;
                std::cout << node_id << std::endl;
                exit(0);
            }
            sub = line.substr(split_point + 1);
            int content_id=id_map.get_content_id(ReadSentence(sub, &d));
            vc_edgelist.push_back(Edge{node_id,content_id});
        }
    }

    void InitUnigramTable() {
        vv_unitable.resize(vv_table_size);
        long long normalizer = 0;
        double d1, power = 0.75;
        for (int a = 0; a < node_count; a++) normalizer += pow(vv_vtou_graph[a].size(), power);
        int i = 0;
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
        std::uniform_int_distribution<> dis(0, node_count - 1);
        while (i < sample_size) {
            unsigned nv = dis(*cnn::rndeng);
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

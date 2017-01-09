//
// Created by Adoni1203 on 16/7/19.
// VV: vertex to vertex
// VC: vertex to content
//

#ifndef WEIBONETEMBEDDING_GRAPH_DATA_H
#define WEIBONETEMBEDDING_GRAPH_DATA_H

#include "dynet/dynet.h"
#include "dynet/dict.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <unordered_set>
#include <boost/algorithm/string.hpp>

using namespace std;

typedef vector<int> SENT_TYPE; //保存句子信息
typedef vector<SENT_TYPE> CONTENT_TYPE; //保存content信息

struct Edge {
    int u_id;
    int v_id;
    int edge_type;
};

struct Node {
    bool with_content;
    CONTENT_TYPE content;
    int embedding_id;
};


struct NetworkData {
    dynet::Dict node_id_map;

    int edge_type_count;
    int edge_count;
    int node_count;
    int normal_node_count;

    int table_size;

    vector<vector<unordered_set<int>>> utov_graph; //保存以u为中心点的vv邻接表
    vector<vector<unordered_set<int>>> vtou_graph; //保存以v为中心点的v
    vector<Node> node_list;
    vector<Edge> edge_list; //保存所有vv边

    vector<vector<int>> u_uni_tables;
    vector<vector<int>> v_uni_tables;

    vector<vector<int>> relation_negative_table;


    explicit NetworkData(vector<string> node_list_file_names, vector<string> edge_list_file_names,
                         string content_file_name,
                         dynet::Dict &d) {

//        cout << "Node list file: " << content_file_name << endl;
//        cout << "Edge list files: " << edge_list_file_name << endl;
//        cout << "Content file: " << content_file_name << endl;


        read_node_list_from_file(node_list_file_names);
        read_edge_list_from_file(edge_list_file_names);
        read_content_from_file(content_file_name, d);

        {
            int next_embedding_id = 0;
            for (int node_id = 0; node_id < node_count; node_id++) {
                if (!node_list[node_id].with_content) {
                    node_list[node_id].embedding_id = next_embedding_id;
                    next_embedding_id++;
                } else {
                    node_list[node_id].embedding_id = -1;
                }
            }
            normal_node_count = next_embedding_id;
        }
        table_size = 1e8;
        InitUniTables(u_uni_tables, utov_graph);
        InitUniTables(v_uni_tables, vtou_graph);
        find_coexist_edge_type();
    }

    void read_node_list_from_file(vector<string> node_list_file_names) {
        cout << "Reading node list ..." << endl;
        node_count = 0;
        for (auto node_list_file_name:node_list_file_names) {
            ifstream file_in(node_list_file_name);
            assert(file_in);
            int part_node_count;
            file_in >> part_node_count;
            node_count += part_node_count;
            node_list.resize(node_count);
            for (int i = 0; i < part_node_count; i++) {
                string node;
                file_in >> node;
                int node_id = node_id_map.convert(node);
                assert(node_id == (node_count - part_node_count + i));
                node_list[node_id].with_content = false;
            }
            file_in.close();
        }
        node_id_map.freeze();
    }

    void read_edge_list_from_file(vector<string> edge_list_file_names) {
        cout << "Reading edge list ..." << endl;
        string line;

        for (auto edge_list_file_name:edge_list_file_names) {
            ifstream file_in(edge_list_file_name);
            assert(file_in);
            int part_edge_count;
            file_in >> part_edge_count >> edge_type_count;
            edge_count += part_edge_count;
            if (edge_type_count > utov_graph.size()) {
                utov_graph.resize(edge_type_count);
                vtou_graph.resize(edge_type_count);
                for (int i = 0; i < edge_type_count; i++) {
                    utov_graph[i].resize(node_count);
                    vtou_graph[i].resize(node_count);
                }
            }
            for (int i = 0; i < part_edge_count; i++) {
                string u, v;
                int edge_type;
                file_in >> u >> v >> edge_type;
                int u_id = node_id_map.convert(u);
                int v_id = node_id_map.convert(v);
                this->edge_list.push_back(Edge{u_id, v_id, edge_type});
                utov_graph[edge_type][u_id].insert(v_id);
                vtou_graph[edge_type][v_id].insert(u_id);
            }
            file_in.close();
        }

    }

    void read_content_from_file(string content_file_name, dynet::Dict &d) {
        cout << "Reading content ..." << endl;
        string line;
        ifstream file_in(content_file_name);
        assert(file_in);
        while (getline(file_in, line)) {
            string::size_type split_pos = line.find(' ');
            string sub = line.substr(0, split_pos - 0);
            int node_id = node_id_map.convert(sub);
            string::size_type start_pos = split_pos + 1;
            CONTENT_TYPE content;
            while (1) {
                string::size_type end_pos = line.find("||||", start_pos + 1);
                if (end_pos == string::npos) {
                    break;
                }
                string sentence_string = line.substr(start_pos, end_pos - start_pos);
                content.push_back(read_sentence(sentence_string, d));
                start_pos = end_pos + 4;
            }
            assert(content.size() > 0);
            node_list[node_id].content = content;
            node_list[node_id].with_content = true;
        }
        file_in.close();
    }


    void find_coexist_edge_type() {
        std::sort(edge_list.begin(), edge_list.end(), [](Edge a, Edge b) {
            if (a.u_id != b.u_id) { return a.u_id < b.u_id; }
            if (a.v_id != b.v_id) { return a.v_id < b.v_id; }
            return a.edge_type<b.edge_type;
        });
        bool could_coexist[edge_type_count][edge_type_count];
        for (int e1=0;e1<edge_type_count;e1++){
            for (int e2=0;e2<edge_type_count;e2++){
                could_coexist[e1][e2]=false;
            }
        }
        auto begin=0;
        while(begin<edge_count){
            int end=begin;
            while(end<edge_count && edge_list[end].u_id==edge_list[begin].u_id && edge_list[end].v_id==edge_list[begin].v_id){
                end++;
            }
            vector<int> coexist_edge_type;
            for(int i=begin;i<end;i++){
                coexist_edge_type.push_back(edge_list[i].edge_type);
            }
            for(auto e1:coexist_edge_type){
                for(auto e2:coexist_edge_type){
                    could_coexist[e1][e2]=true;
                }
            }
            begin=end;
        }
        relation_negative_table.resize(edge_type_count);
        for(int e1=0;e1<edge_type_count;e1++){
            relation_negative_table[e1].resize(0);
            for (int e2=0;e2<edge_type_count;e2++){
                if (e1!=e2 && could_coexist[e1][e2]){
                    relation_negative_table[e1].push_back(e2);
                }
            }
        }
    }

    void InitUniTables(vector<vector<int>> &tables, vector<vector<unordered_set<int>>> &graph) {
        tables.resize(edge_type_count);
        for (int edge_type = 0; edge_type < edge_type_count; edge_type++) {
            cout << "Initializing uni table ..." << edge_type << endl;
            tables[edge_type].resize(table_size);
            long long normalizer = 0;
            double d1, power = 0.75;
            for (int node_id = 0; node_id < node_count; node_id++) {
                normalizer += pow(graph[edge_type][node_id].size(), power);
            }
            cout << "normalizer: " << normalizer << endl;
            int i = 0;
            d1 = pow(graph[edge_type][i].size(), power) / (double) normalizer;
            for (int a = 0; a < table_size; a++) {
                while (a / (double) table_size >= d1) {
                    i++;
                    d1 += pow(graph[edge_type][i].size(), power) / (double) normalizer;
                }
                assert(i < node_count);
                tables[edge_type][a] = i;

            }
        }
    }

    int relation_type(int u_id, int v_id, int edge_type) {
        if (utov_graph[edge_type][u_id].find(v_id) != utov_graph[edge_type][u_id].end()) {
            return 1;
        }
        return 0;
    }


    void u_id_neg_sample(int sample_size, Edge edge, vector<Edge> &neg_edges) {
        int i = 0;
        uniform_int_distribution<> dis(0, table_size - 1);
        while (i < sample_size) {
            int neg_u_id = u_uni_tables[edge.edge_type][dis(*dynet::rndeng)];
            assert(neg_u_id < node_count);
            if (neg_u_id == edge.u_id || neg_u_id == edge.v_id ||
                relation_type(neg_u_id, edge.v_id, edge.edge_type) == 1)
                continue;
            neg_edges.push_back(Edge{neg_u_id, edge.v_id, edge.edge_type});
            i++;
        }
    }

    void v_id_neg_sample(int sample_size, Edge edge, vector<Edge> &neg_edges) {
        int i = 0;
        uniform_int_distribution<> dis(0, table_size - 1);
        while (i < sample_size) {
            int neg_v_id = v_uni_tables[edge.edge_type][dis(*dynet::rndeng)];
            assert(neg_v_id < node_count);
            if (neg_v_id == edge.u_id || neg_v_id == edge.v_id ||
                relation_type(edge.u_id, neg_v_id, edge.edge_type) == 1)
                continue;
            neg_edges.push_back(Edge{edge.u_id, neg_v_id, edge.edge_type});
            i++;
        }
    }


    void edge_type_neg_sample(Edge edge, vector<Edge> &neg_edges) {
        for(auto possible_neg_edge_type:relation_negative_table[edge.edge_type]){
            if (relation_type(edge.u_id, edge.v_id, possible_neg_edge_type) == 1)
                continue;
            neg_edges.push_back(Edge{edge.u_id, edge.v_id, possible_neg_edge_type});
        }
    }

    void v_id_and_edge_type_neg_sample(int sample_size, Edge edge, vector<Edge> &neg_edges) {
        int i = 0;
        uniform_int_distribution<> dis1(0, table_size - 1);
        uniform_int_distribution<> dis2(0, edge_type_count - 1);

        while (i < sample_size) {
            int neg_v_id = u_uni_tables[edge.edge_type][dis1(*dynet::rndeng)];
            int neg_edge_type = dis2(*dynet::rndeng);
            assert(neg_v_id < node_count);
            if (neg_v_id == edge.u_id || neg_v_id == edge.v_id ||
                relation_type(edge.u_id, neg_v_id, neg_edge_type) == 1)
                continue;
            neg_edges.push_back(Edge{edge.u_id, neg_v_id, neg_edge_type});
            i++;
        }
    }

    vector<Edge> edge_neg_sample(int sample_size, Edge edge) {
        vector<Edge> neg_edges;
        v_id_neg_sample(sample_size, edge, neg_edges);
        edge_type_neg_sample(edge, neg_edges);
        v_id_and_edge_type_neg_sample(sample_size, edge, neg_edges);
        return neg_edges;
    }
};

#endif //WEIBONETEMBEDDING_GRAPH_DATA_H

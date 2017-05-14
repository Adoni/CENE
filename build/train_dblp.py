import subprocess


def main():
    command = [
        './dlne_main',
        '--dynet-mem 1500',
        '--dynet-seed 1',
        '--node_list_file ~/dblp_data/nodes_0.data',
        '~/dblp_data/nodes_1.data',
        '--edge_list_file ~/dblp_data/edge_list_0.data',
        '~/dblp_data/edge_list_1.data',
        '--content_node_file ~/dblp_data/contents.data',
        '--word_embedding_file ~/DLNE_data/2dblp/embedding.data',
        '--to_be_saved_index_file_name ~/dblp_data/to_be_saved_index.data',
        '--relation_matrix_file ~/dblp_data/relation_matrix.data',
        '--params_eta0 0.02',
        '--params_eta_decay 0.001',
        '--lookup_params_eta0 0.02',
        '--lookup_params_eta_decay 0.001',
        '--workers 20',
        '--iterations 500',
        '--batch_size 1000',
        '--save_every_i 10',
        '--report_every_i 10000',
        '--update_epoch_every_i 10000',
        '--negative 15 25',
        '--alpha 1 1',
        '--beta 0.0',
        '--lambda 1e-6',
        '--embedding_method WordAvg',
        '--score_function 0',
    ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()

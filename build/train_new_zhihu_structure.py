import subprocess


def main():
    command = [
        './dlne_main',
        '--dynet-mem 1500',
        '--dynet-seed 12345',
        '--node_list_file ~/zhihu_data/83095_2/nodes_0.data',
        '--edge_list_file ~/zhihu_data/83095_2/edge_list_0.data',
        '--content_node_file ~/zhihu_data/83095_2/content_empty.data',
        '--word_embedding_file ~/zhihu_data/83095_2/embedding.data',
        '--to_be_saved_index_file_name ~/zhihu_data/83095_2/to_be_saved_index.data',
        '--relation_matrix_file ~/zhihu_data/83095_2/relation_matrix.data',
        '--params_eta0 0.03',
        '--params_eta_decay 0.003',
        '--lookup_params_eta0 0.03',
        '--lookup_params_eta_decay 0.003',
        '--workers 15',
        '--iterations  50',
        '--batch_size 100',
        '--save_every_i 1',
        '--report_every_i 100000',
        '--update_epoch_every_i 10000',
        '--negative 15 15 15 15 15',
        '--alpha 1 1 1 1 1',
        '--embedding_method WordAvg',
        '--score_function 2',
    ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()

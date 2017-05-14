import subprocess


def main():
    command = [
        './dlne_main',
        '--dynet-mem 7000',
        '--dynet-seed 12345',
        '--node_list_file ~/zhihu_data/83095_3/nodes_0.data',
        '~/zhihu_data/83095_3/nodes_1.data',
        '--edge_list_file ~/zhihu_data/83095_3/edge_list_0.data',
        '~/zhihu_data/83095_3/edge_list_1.data',
        '~/zhihu_data/83095_3/edge_list_2.data',
        '~/zhihu_data/83095_3/edge_list_3.data',
        '~/zhihu_data/83095_3/edge_list_4.data',
        '--content_node_file ~/zhihu_data/83095_3/contents.data',
        '--word_embedding_file ~/zhihu_data/83095_3/embedding256.data',
        '--to_be_saved_index_file_name ~/zhihu_data/83095_3/to_be_saved_index.data',
        '--relation_matrix_file ~/zhihu_data/83095_3/relation_matrix.data',
        '--params_eta0 0.01',
        '--params_eta_decay 0.003',
        '--lookup_params_eta0 0.01',
        '--lookup_params_eta_decay 0.003',
        '--workers 20',
        '--iterations  50',
        '--batch_size 1000',
        '--save_every_i 1',
        '--report_every_i 100000',
        '--update_epoch_every_i 10000',
        '--negative 10 10',
        '--alpha 1 1',
        '--beta 0.6',
        '--lambda 1e-6',
        '--embedding_method WordAvg',
        '--score_function 0',
    ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()

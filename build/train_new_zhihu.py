import subprocess


def main():
    command = [
        './dlne_main',
        '--dynet-mem 5000',
        '--dynet-seed 12345',
        '--node_list_file ~/zhihu_data/cikm/data/node_list_0.data',
        '~/zhihu_data/cikm/data/node_list_1.data',
        '--edge_list_file ~/zhihu_data/cikm/data/edge_list_0.data',
        '~/zhihu_data/cikm/data/edge_list_1.data',
        '--content_node_file ~/zhihu_data/cikm/data/contents.data',
        '--word_embedding_file ~/zhihu_data/cikm/data/word_embedding.data',
        '--to_be_saved_index_file_name ~/zhihu_data/cikm/data/to_be_saved_index.data',
        '--params_eta0 0.01',
        '--params_eta_decay 0.003',
        '--workers 20',
        '--iterations  50',
        '--batch_size 1000',
        '--save_every_i 1',
        '--report_every_i 100000',
        '--update_epoch_every_i 10000',
        '--negative 10 10',
        '--alpha 1 1',
        '--lambda 1e-6',
        '--embedding_method WordAvg',
    ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()

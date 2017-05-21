import subprocess


def main():
    command = [
        './dlne_main',
        '--dynet-mem 1500',
        '--dynet-seed 1',
        '--node_list_file ~/m10_data/node_list_0.data',
        '~/m10_data/node_list_1.data',
        '--edge_list_file ~/m10_data/edge_list_0.data',
        '~/m10_data/edge_list_1.data',
        '--content_node_file ~/m10_data/contents.data',
        '--word_embedding_file ~/m10_data/embedding_en.data',
        '--to_be_saved_index_file_name ~/m10_data/to_be_saved_index.data',
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
        '--negative 15 25 25',
        '--alpha 1 1 1',
        '--lambda 1e-6',
        '--embedding_method WordAvg',
    ]
    command = ' '.join(command)
    print(command)
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()

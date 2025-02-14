from datasets import load_dataset

if __name__ == "__main__":

    dataset = load_dataset(
        '/mnt/matylda6/isedlacek/projects/huggingface_asr/src/dataset_builders/spokenwoz',
        data_dir='/mnt/matylda4/kesiraju/datasets/dialogue_datasets/SpokenWoz_2023',
        mode='multiwoz',
        #asr_json_dir='/mnt/matylda4/hegde/int_ent/spokenwoz_whisper_transcripts/data_processed',
        num_proc=16,
        splits=['train', 'dev', 'test'],
        title_case_slots=True,
        trust_remote_code=True,
    )

    dataset = dataset.remove_columns(['audio'])

    print(dataset)
    dataset.save_to_disk('/mnt/scratch/tmp/isedlacek/data/spokenwoz_alibaba', num_proc=4)

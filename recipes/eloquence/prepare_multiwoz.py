from datasets import load_dataset

if __name__ == "__main__":

    dataset = load_dataset(
        '/mnt/matylda6/isedlacek/projects/huggingface_asr/src/dataset_builders/sa_multiwoz',
        data_dir='/mnt/matylda4/kesiraju/code/OLMo/speech_aware_dialogue/processed',
        num_proc=16,
        #splits=['dev', 'train'],
        title_case_slots=True,
        trust_remote_code=True,
    )

    print(dataset)
    dataset.save_to_disk('/mnt/scratch/tmp/isedlacek/data/sa_multiwoz_tests', num_proc=4)

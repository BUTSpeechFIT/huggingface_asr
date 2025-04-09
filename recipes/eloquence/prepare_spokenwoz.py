from datasets import load_dataset

# TODO: modify these paths based on your setup
REPO_ROOT = '/mnt/matylda6/isedlacek/projects/huggingface_asr'
DATA_DIR = '/mnt/matylda4/kesiraju/datasets/dialogue_datasets/SpokenWoz_2023' # path to the downloaded spokenwoz dataset
OUT_DIR = '/mnt/scratch/tmp/isedlacek/data/spokenwoz'

if __name__ == "__main__":

    dataset = load_dataset(
        REPO_ROOT + '/src/dataset_builders/spokenwoz', # path to the dataset builder
        data_dir=DATA_DIR,
        mode='multiwoz', # multiwoz-compatible default spokenwoz creation config
        #asr_json_dir='/mnt/matylda4/hegde/int_ent/spokenwoz_whisper_transcripts/data_processed',
        num_proc=16,
        splits=['train', 'dev', 'test'],
        title_case_slots=True,
        trust_remote_code=True,
    )
    print(dataset)


    # Save the dataset to disk
    dataset.save_to_disk(OUT_DIR, num_proc=4)

from Models.Encoder.data_objects.random_cycler import RandomCycler
from Models.Encoder.data_objects.speaker_batch import SpeakerBatch
from Models.Encoder.data_objects.speaker import Speaker
from Models.Encoder.params_data import partials_n_frames
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from pathlib import Path

# TODO: improve with a pool of speakers for data efficiency

class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets_root: Path, split: str = "train", val_split = 0.2):
        self.root = datasets_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        train_speakers, val_speakers = train_test_split(
            speaker_dirs, test_size=val_split, random_state=42)

        if split == "train":
            selected_speakers = train_speakers
        elif split == "val":
            selected_speakers = val_speakers
        else:
            raise ValueError("Invalid split value. Use 'train' or 'val'.")

        self.speakers = [Speaker(speaker_dir) for speaker_dir in selected_speakers]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
    
class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames) 
    
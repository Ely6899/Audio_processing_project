import os
from pathlib import Path
import typing
import re

from tqdm import tqdm

from ConstPaths import RavdessPaths

from audio_dataset import RavdessRawDataWithNeutral

STATEMENT_MAPPING = {
    "kids": "Kids are talking by the door",
    "dogs": "Dogs are sitting by the door",
}

def get_neutral_files_from_ravdess() -> typing.Set[Path]:
    all_neutral_files: typing.Set[Path] = set(RavdessPaths.AUDIO_FILES_DATA.rglob("*-*-01-*-*-*-*.wav"))
    return all_neutral_files


def get_actor_number_value(file_path: Path) -> typing.Tuple[str, int]:
    actor_number, actor_value = RavdessRawDataWithNeutral.get_attribute_from_filename(file_path, "actor")
    return actor_number, actor_value


def get_statement_number_value(file_path: Path) -> typing.Tuple[str, str]:
    statement_number, statement_value = RavdessRawDataWithNeutral.get_attribute_from_filename(file_path, "statement")
    return statement_number, statement_value


def get_repetition_number_value(file_path: Path) -> typing.Tuple[str, int]:
    repetition_number, repetition_value = RavdessRawDataWithNeutral.get_attribute_from_filename(file_path, "repetition")
    return repetition_number, repetition_value


def fetch_data_for_deepfake() -> None:
    neutral_files_paths: typing.Set[Path] = get_neutral_files_from_ravdess()
    for file_path in tqdm(neutral_files_paths, "Saving txt files for deepfake_process"):
        actor_number, actor_value = get_actor_number_value(file_path)
        actor_path: Path = Path(os.path.join(RavdessPaths.SYNTH_NEUTRAL_PATH, f"Actor_{actor_number}"))

        os.makedirs(actor_path, exist_ok=True)

        statement_number, statement_value = get_statement_number_value(file_path)
        repetition_number, repetition_value = get_repetition_number_value(file_path)

        saved_file_path: Path = (Path(os.path.join(actor_path, f"{statement_value}_rep{repetition_value}_act{actor_value}"))
                           .with_suffix(".txt"))

        original_name = file_path.name
        def change_fifth_number_in_file_name(filename: str, file_statement_num: str):
            # Define the regex pattern to match the middle number
            pattern = r'(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2}).wav'

            # Define a function to swap '01' and '02' in the middle part
            def replace_middle(match):
                first, second, third, fourth, file_statement, sixth, seventh = match.groups()
                # Swap '01' with '02' and vice versa
                if file_statement_num == '01':
                    file_statement = '02'
                elif file_statement_num == '02':
                    file_statement = '01'
                return f"{first}-{second}-{third}-{fourth}-{file_statement}-{sixth}-{seventh}.wav"

            # Use re.sub with the replace function
            new_filename = re.sub(pattern, replace_middle, filename)
            return new_filename

        matching_neutral_file_name = change_fifth_number_in_file_name(original_name, statement_number)
        neutral_audio_path_reference: Path = file_path.with_name(matching_neutral_file_name)

        with open(saved_file_path, 'w') as saved_file:
            saved_file.writelines([STATEMENT_MAPPING.get(statement_value), '\n', neutral_audio_path_reference.__str__()])

if __name__ == '__main__':
    fetch_data_for_deepfake()

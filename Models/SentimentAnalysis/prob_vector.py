
import pandas as pd
from sklearn.calibration import LabelEncoder
import torch
import tqdm
from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS
from Preprocess import audio_to_mel_spectrogram
from audio_dataset import AllRawData, EmotionSpecDataset, RavdessRawData
from models import ResNetWithAttention
from pprint import pprint

def get_sample_probabilities_of_model(model, mel_spec_tensor):
    """
    Get the probabilities from the model for a tensor of a single sample.
    """ 
    model.eval()
    # Add batch dimension since model expects input in shape [B, C, H, W]
    if mel_spec_tensor.dim() == 3:
        # If input_tensor is 3D, add a batch dimension
        mel_spec_tensor = mel_spec_tensor.unsqueeze(0)
    
    # Move input and model to the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mel_spec_tensor = mel_spec_tensor.to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        logits = model(mel_spec_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        # predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # turn from 2d to 1d
    probabilities = probabilities.squeeze(0)  # Remove batch dimension if present
    return probabilities.cpu().numpy()

def get_dataset_probabilities_of_model(model, dataset):
    """
    Get the probabilities from the model for the given dataset.
    returns a dataframe with probabilities for each sample in the dataset.
    """
    all_probabilities = []
    for sample in tqdm.tqdm(dataset, desc="Processing samples", unit="sample"):
        # Get the probabilities for each sample
        all_probabilities.append(get_sample_probabilities_of_model(model, sample))
    
    # Convert probabilities to a DataFrame
    probabilities_df = pd.DataFrame(all_probabilities, columns=[f"Class_{i}" for i in range(all_probabilities[0].shape[0])])
    
    # Add predicted class (class with highest probability)
    probabilities_df['predicted_class'] = probabilities_df.iloc[:, :all_probabilities[0].shape[0]].idxmax(axis=1)
    probabilities_df['predicted_class'] = probabilities_df['predicted_class'].str.replace('Class_', '').astype(int)
    
    # Add predicted probability value (maximum probability)
    probabilities_df['predicted_probability'] = probabilities_df.iloc[:, :all_probabilities[0].shape[0]].max(axis=1)
    return probabilities_df

def preproccess_like_in_dataloader(file_path):
    # noam: audio_to_mel_spectrogram returns shape (freq_bins, time_frames)
    mel_spectrogram = audio_to_mel_spectrogram(file_path=file_path, max_length_in_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS)

    # Convert to torch.Tensor
    mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
    
    # Now expand to shape (1, freq_bins, time_frames)
    mel_spectrogram = mel_spectrogram.unsqueeze(dim=0)
    
    return mel_spectrogram

def get_raw_sample_attributes(model, raw_data_sample):
    """
    Get the probabilities from the model for a single sample from audio raw data sample
    
    Args:
        model: The trained model to use for prediction.
        raw_data_sample: (Path, label) tuple where Path is the path to the audio file and label is the emotion label.
    Returns:
        attributes: A dictionary containing the predicted class and its probability.  
            probabilities of each label
            predicted label 
            predicted probability 
            true label 
            path 
    """
    attr = {} # dictionary to hold attributes
    path, true_label = raw_data_sample
    
    attr["path"] = path
    attr["true_label"] = true_label
    
    # preper data to the model
    tensor_mel_spec = preproccess_like_in_dataloader(path)
    # insert data to model and get probabilities
    probabilities = get_sample_probabilities_of_model(model, tensor_mel_spec)
    
    # get the de/encoding from classes(nums 0-7 for model) to labels(strings)
    labels = ['angry',
            'calm',
            'disgust',
            'fearful',
            'happy',
            'neutral',
            'sad',
            'surprised']
    label_encoder = LabelEncoder()
    label_encoder.fit(labels) # order Doesn't matter!
    
    # insert all the probablities as such: "class label" : <probability_for_that_class>
    # Add probabilities for each class to attributes
    for i, prob in enumerate(probabilities):
        class_label = label_encoder.classes_[i]
        attr[f'prob {class_label}'] = float(prob)  # Convert numpy float to Python float for better serialization

    # Add predicted class and its probability
    predicted_class_idx = probabilities.argmax()
    predicted_class_label = label_encoder.classes_[predicted_class_idx]
    attr["predicted_label"] = predicted_class_label
    attr["predicted_probability"] = float(probabilities[predicted_class_idx])

    # Convert the dictionary to a pandas Series with a custom order
    # Define the order of attributes in the Series
    attr_order = [
        'path', 
        'true_label', 
        'predicted_label', 
        'predicted_probability',
        'prob angry', 
        'prob calm', 
        'prob disgust', 
        'prob fearful', 
        'prob happy', 
        'prob neutral', 
        'prob sad', 
        'prob surprised'
    ]
    
    # Create a Series with the specified order WARNING: This will only include keys that are present in attr_order
    attr_series = pd.Series({k: attr[k] for k in attr_order if k in attr})
    
    return attr_series

def get_raw_dataset_attributes(model, raw_dataset: set):
    """
    Get the attributes for each sample in the raw dataset.
    
    Args:
        model: The trained model to use for prediction.
        raw_dataset: An iterable of raw data samples, where each sample is a (Path, label) tuple.
        
    Returns:
        attributes_list: A list of dictionaries, each containing attributes for a sample.
    """
    attributes_list = []
    for sample in tqdm.tqdm(raw_dataset, desc="Processing samples", unit="sample"):
        attributes = get_raw_sample_attributes(model, sample)
        attributes_list.append(attributes)
    
    # Convert the list of Series objects to a DataFrame
    attributes_df = pd.DataFrame(attributes_list)
    
    return attributes_df

def tests1():
    ######### Probability Vector Dataframe #########

    # raw data loading:
    ravdess_raw_data = RavdessRawData(include_calm=True, include_aug=False)

    ravdess_raw_data.print_all_label_counts()

    all_data = AllRawData((ravdess_raw_data, ))

    train_set, val_set = all_data.train_val_test_split(0.2)

    # pytorch's dataset creation:

    train_ds = EmotionSpecDataset(train_set)
    val_ds = EmotionSpecDataset(val_set)



    # create the model:

    model = ResNetWithAttention(num_classes=8)

    # Load the model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("Eli ResNetWithAttention.pt", map_location=device)
    
    """
    Get the probabilities from the model for the given dataset.
    """
    # probabilities = get_sample_probabilities_of_model(model, train_ds[0])  # Get probabilities for the first sample in the training dataset

    # Use a subset of the training dataset for quick testing
    subset_size = 10000  # Adjust this number based on how small a sample you want
    ds_subset = torch.utils.data.Subset(val_ds, range(subset_size))

    print(f"Computing probabilities for {subset_size} samples instead of {len(val_ds)} total samples")
    probabilities_df = get_dataset_probabilities_of_model(model, ds_subset)
    # print(f"True label: {label}")
    # print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities_df}")
    
    # Save the DataFrame to CSV
    filepath = "probabilities.csv"
    probabilities_df.to_csv(filepath, index=False)
    print(f"Saved probabilities to {filepath}")

if __name__ == "__main__":
    ######### Probability Vector Dataframe #########

    # raw data loading:
    ravdess_raw_data = RavdessRawData(include_calm=True, include_aug=False)

    ravdess_raw_data.print_all_label_counts()

    all_raw_data = AllRawData((ravdess_raw_data, ))
    
    data = set(list(all_raw_data.all_data)[:100])
    
    # create the model:
    model = ResNetWithAttention(num_classes=8)

    # Load the model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("Eli ResNetWithAttention.pt", map_location=device)
    
    # Get the attributes for the sample
    attributes = get_raw_dataset_attributes(model, data)
    pprint(attributes[:4])
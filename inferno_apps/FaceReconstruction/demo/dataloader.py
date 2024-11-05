from dataclasses import dataclass
import json
import random
import time
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import v2 as transforms
from torchvision.io import read_image


class Transform(object):
    def __init__(self, img_size=256):
        self.img_size = img_size

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToDtype(torch.float32, scale=True),
                normalize,
            ]
        )

    def __call__(self, img):
        img = self.transform(img)
        return img


idx2emotion = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Neutral",
    6: "Sadness",
    7: "Surprise",
}

emotion2idx = {
    "Anger": 0,
    "Contempt": 1,
    "Disgust": 2,
    "Fear": 3,
    "Happiness": 4,
    "Neutral": 5,
    "Sadness": 6,
    "Surprise": 7,
}


@dataclass
class BioVidDatasetConfig:
    path_to_frame_labels: str
    path_to_video_frame: str
    max_length: int = 256
    img_size: int = 256
    load_au_features: bool = False
    load_emotion_labels: bool = True
    load_stimulus_values: bool = True
    load_stimulus_label: bool = True
    load_pspi_no_au43: bool = True


class BioVidDataset(Dataset):
    def __init__(self, data_config: BioVidDatasetConfig):

        self.video_frame_dir = path_to_video_frame
        self.frame_labels_dir = path_to_frame_labels

        self.video_names = os.listdir(self.video_frame_dir)

        self.load_au_features = data_config.load_au_features
        self.load_emotion_labels = data_config.load_emotion_labels
        self.load_stimulus_values = data_config.load_stimulus_values
        self.load_stimulus_label = data_config.load_stimulus_label
        self.load_pspi_no_au43 = data_config.load_pspi_no_au43

        self.video_chunks = []
        self.max_length = data_config.max_length

        self._transform = Transform(img_size=data_config.img_size)

        for video_name in self.video_names:

            random_start = random.randint(0, data_config.max_length)

            video_length = len(
                os.listdir(os.path.join(self.video_frame_dir, video_name))
            )

            for chunk_start_pointer in range(
                random_start, video_length + 1, self.max_length
            ):
                if chunk_start_pointer + self.max_length > video_length:
                    continue
                self.video_chunks.append(
                    (
                        video_name,
                        chunk_start_pointer,
                        min(chunk_start_pointer + self.max_length, video_length),
                    )
                )

    def __len__(self):
        return len(self.video_chunks)

    def __getitem__(self, idx):

        # print(f"Loading video chunk {idx}")

        # start_time = time.time()
        video_name, start_frame_id, end_frame_id = self.video_chunks[idx]

        frames = []
        au_features = []
        emotion_labels = []
        temperature_values = []
        stimulus_values = []
        pspi_no_au43 = []

        for frame_id in range(start_frame_id, end_frame_id):
            frame_name = os.path.join(
                self.video_frame_dir, video_name, f"frame_{frame_id}.jpg"
            )

            frame = read_image(frame_name)
            frame = self._transform(frame)
            frames.append(frame.unsqueeze(0))

            if (
                self.load_au_features
                or self.load_emotion_labels
                or self.load_stimulus_values
                or self.load_stimulus_label
                or self.load_pspi_no_au43
            ):

                frame_label_name = os.path.join(
                    self.frame_labels_dir, video_name, f"{frame_id}.json"
                )
                with open(frame_label_name, "r") as f:
                    frame_label = json.load(f)

                if self.load_au_features:
                    au_feature = frame_label["au_bp4d_score"]
                    au_feature.extend(frame_label["au_disfa_score"])
                    au_features.append(au_feature)

                if self.load_emotion_labels:
                    emotion_labels.append(emotion2idx[frame_label["emotion"]])

                if self.load_stimulus_values:
                    temperature_values.append(frame_label["temperature"])

                if self.load_stimulus_label:
                    stimulus_values.append(frame_label["label"])

                if self.load_pspi_no_au43:
                    pspi_no_au43.append(frame_label["pspi_no_au43"])

        if len(frames) > 0:
            frames = torch.cat(frames, dim=0)

        if len(au_features) > 0:
            au_features = torch.tensor(au_features)

        if len(emotion_labels) > 0:
            emotion_labels = torch.tensor(emotion_labels)

        if len(temperature_values) > 0:
            temperature_values = torch.tensor(temperature_values)

        if len(stimulus_values) > 0:
            stimulus_values = torch.tensor(stimulus_values)

        if len(pspi_no_au43) > 0:
            pspi_no_au43 = torch.tensor(pspi_no_au43)

        # print(f"Time taken to load video chunk {idx}: {time.time() - start_time}")

        return (
            frames,
            au_features,
            emotion_labels,
            temperature_values,
            stimulus_values,
            pspi_no_au43,
        )


if __name__ == "__main__":

    path_to_frame_labels = "/media/tien/SSD-NOT-OS/processed_pain_data_no_facedetector/"
    path_to_video_frame = (
        "/media/tien/SSD-DATA/data/BioVid HeatPain DB/PartC/extracted_frame/"
    )

    def assert_sample(sample):
        assert sample[0].size(0) == 256
        assert sample[1].size(0) == 256
        assert sample[2].size(0) == 256
        assert sample[3].size(0) == 256
        assert sample[4].size(0) == 256
        assert sample[5].size(0) == 256

    dataset = BioVidDataset(
        path_to_frame_labels,
        path_to_video_frame,
        max_length=256,
        img_size=224,
        load_au_features=True,
        load_emotion_labels=True,
        load_stimulus_values=True,
        load_stimulus_label=True,
        load_pspi_no_au43=True,
    )
    print(len(dataset))

    for i in range(10):
        random_idx = random.randint(0, len(dataset))

        sample = dataset[random_idx]

        assert_sample(sample)

    assert_sample(dataset[0])
    assert_sample(dataset[-1])

    print("Test passed")

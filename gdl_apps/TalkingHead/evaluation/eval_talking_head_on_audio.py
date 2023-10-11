from gdl_apps.TalkingHead.evaluation.eval_lip_reading import TalkingHeadWrapper, dict_to_device, save_video
from pathlib import Path
import librosa
import numpy as np
from gdl.utils.collate import robust_collate
import torch
import os, sys
from gdl.utils.PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer
from tqdm.auto import tqdm
from gdl.datasets.AffectNetAutoDataModule import AffectNetExpressions
import trimesh
import copy


def create_condition(talking_head, sample, emotions=None, intensities=None, identities=None):
    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_label', False): # mead GT expression label
        if emotions is None:
            emotions = [AffectNetExpressions.Neutral.value]
        sample["gt_expression_label_condition"] = torch.nn.functional.one_hot(torch.tensor(emotions), 
            num_classes=talking_head.get_num_emotions()).numpy()

    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_intensity', False): # mead GT expression intensity
        if intensities is None:
            intensities = [2]
        sample["gt_expression_intensity_condition"] = torch.nn.functional.one_hot(torch.tensor(intensities), 
            num_classes=talking_head.get_num_intensities()).numpy()

    if talking_head.cfg.model.sequence_decoder.style_embedding.get('gt_expression_identity', False): 
        if identities is None:
            identities = [0]
        sample["gt_expression_identity_condition"] = torch.nn.functional.one_hot(torch.tensor(identities), 
            num_classes=talking_head.get_num_identities()).numpy()
    return sample


def interpolate_condition(sample_1, sample_2, length, interpolation_type="linear"): 
    keys_to_interpolate = ["gt_expression_label_condition", "gt_expression_intensity_condition", "gt_expression_identity_condition"]

    sample_result = copy.deepcopy(sample_1)
    for key in keys_to_interpolate:
        condition_1 = sample_1[key]
        condition_2 = sample_2[key]

        # if temporal dimension is missing, add it
        if len(condition_1.shape) == 1:
            condition_1 = np.expand_dims(condition_1, axis=0)
            # condition_1 = condition_1.unsqueeze(0)
        if len(condition_2.shape) == 1:
            condition_2 = np.expand_dims(condition_2, axis=0)
            # conditions_2 = condition_2.unsqueeze(0)

        
        # if temporal dimension 1, repeat it
        if condition_1.shape[0] == 1:
            condition_1 = condition_1.repeat(length, axis=0)
            # condition_1 = condition_1.repeat(length)
        if condition_2.shape[0] == 1:
            condition_2 = condition_2.repeat(length, axis=0)
            # condition_2 = condition_2.repeat(length)

        # interpolate
        if interpolation_type == "linear":
            # interpolate from condition_1 to condition_2 along the length
            weights = np.linspace(0, 1, length)[..., np.newaxis]
        elif interpolation_type == "nn":
            # interpolate from condition_1 to condition_2 along the length
            weights = np.linspace(0, 1, length)[..., np.newaxis]
            weights = np.round(weights)
        else:
            raise ValueError(f"Unknown interpolation type {interpolation_type}")
        
        interpolated_condition = condition_1 * (1 - weights) + condition_2 * weights
        sample_result[key] = interpolated_condition

    return sample_result


def eval_talking_head_on_audio(talking_head, audio_path, silent_frames_start=0, silent_frames_end=0, 
    silent_emotion_start = 0, silent_emotion_end = 0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    talking_head = talking_head.to(device)
    # talking_head.talking_head_model.preprocessor.to(device) # weird hack
    sample = create_base_sample(talking_head, audio_path, silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end)
    # samples = create_id_emo_int_combinations(talking_head, sample)
    samples = create_high_intensity_emotions(talking_head, sample, 
                                             silent_frames_start=silent_frames_start, silent_frames_end=silent_frames_end, 
                                            silent_emotion_start = silent_emotion_start, silent_emotion_end = silent_emotion_end)
    run_evalutation(talking_head, samples, audio_path,  silent_start=silent_frames_start, silent_end=silent_frames_end)
    print("Done")


def create_base_sample(talking_head, audio_path, smallest_unit=1, silent_frames_start=0, silent_frames_end=0):
    wavdata, sampling_rate = read_audio(audio_path)
    sample = process_audio(wavdata, sampling_rate, video_fps=25)
    # pad the audio such that it is a multiple of the smallest unit
    sample["raw_audio"] = np.pad(sample["raw_audio"], (0, smallest_unit - sample["raw_audio"].shape[0] % smallest_unit))
    if silent_frames_start > 0:
        sample["raw_audio"] = np.concatenate([np.zeros((silent_frames_start, sample["raw_audio"].shape[1]), dtype=sample["raw_audio"].dtype), sample["raw_audio"]], axis=0)
    if silent_frames_end > 0:
        sample["raw_audio"] = np.concatenate([sample["raw_audio"], np.zeros((silent_frames_end, sample["raw_audio"].shape[1]), dtype=sample["raw_audio"].dtype)], axis=0)
    T = sample["raw_audio"].shape[0]
    sample["reconstruction"] = {}
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]] = {} 
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_exp"] = np.zeros((T, 50), dtype=np.float32)
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_shape"] = np.zeros((300), dtype=np.float32)
    # sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_shape"] = np.zeros((T, 300), dtype=np.float32)
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_jaw"] = np.zeros((T, 3), dtype=np.float32)
    sample["reconstruction"][talking_head.cfg.data.reconstruction_type[0]]["gt_tex"] = np.zeros((50), dtype=np.float32)
    sample = create_condition(talking_head, sample)
    return sample


def create_name(int_idx, emo_idx, identity_idx, training_subjects): 
    intensity = int_idx
    emotion = emo_idx
    identity = identity_idx

    emotion = AffectNetExpressions(emotion).name
    identity = training_subjects[identity]
    suffix = f"_{identity}_{emotion}_{intensity}"
    return suffix


def create_interpolation_name(start_int_idx, end_int_idx, 
                              start_emo_idx, end_emo_idx, 
                              start_identity_idx, end_identity_idx, 
                              training_subjects, interpolation_type): 
    
    start_emotion = AffectNetExpressions(start_emo_idx).name
    end_emotion = AffectNetExpressions(end_emo_idx).name
    start_identity = training_subjects[start_identity_idx]
    end_identity = training_subjects[end_identity_idx]
    suffix = f"_{start_identity}to{end_identity}_{start_emotion}2{end_emotion}_{start_int_idx}to{end_int_idx}_{interpolation_type}"
    return suffix


def create_id_emo_int_combinations(talking_head, sample):
    samples = []
    training_subjects = talking_head.get_subject_labels('training')
    for identity_idx in range(0, talking_head.get_num_identities()):
        for emo_idx in range(0, talking_head.get_num_emotions()):
            for int_idx in range(0, talking_head.get_num_intensities()):
                sample_copy = copy.deepcopy(sample)
                sample_copy = create_condition(talking_head, sample_copy, 
                                               emotions=[emo_idx], 
                                               identities=[identity_idx], 
                                               intensities=[int_idx])

                sample_copy["output_name"] = create_name(int_idx, emo_idx, identity_idx, training_subjects)

                samples.append(sample_copy)
    return samples


def create_high_intensity_emotions(talking_head, sample, identity_idx=None, emotion_index_list=None, 
                                   silent_frames_start=0, silent_frames_end=0, 
                                   silent_emotion_start = 0, silent_emotion_end = 0):
    samples = []
    training_subjects = talking_head.get_subject_labels('training')
    # for identity_idx in range(0, talking_head.get_num_identities()): 
    identity_idx = identity_idx or 0
    emotion_index_list = emotion_index_list or list(range(0, talking_head.get_num_emotions()))
    for emo_idx in emotion_index_list:
        # for int_idx in range(0, talking_head.get_num_intensities()):
        if emotion_index_list == [0]:
            int_idx = 0
        else:
            int_idx = talking_head.get_num_intensities() - 1
        sample_copy = copy.deepcopy(sample)
        sample_copy = create_condition(talking_head, sample_copy, 
                                        emotions=[emo_idx], 
                                        identities=[identity_idx], 
                                        intensities=[int_idx])
        if silent_frames_start > 0:
            T = sample_copy["raw_audio"].shape[0]
            cond = sample_copy["gt_expression_label_condition"]
            if cond.shape[0] == 1:
                cond = cond.repeat(T, axis=0)
            cond[:silent_frames_start] = 0 
            cond[:silent_frames_start, silent_emotion_start] = 1
            sample_copy["gt_expression_label_condition"]= cond
        if silent_frames_end > 0:
            T = sample_copy["raw_audio"].shape[0]
            cond = sample_copy["gt_expression_label_condition"]
            if cond.shape[0] == 1:
                cond = cond.repeat(T, axis=0)
            cond[-silent_frames_end:] = 0
            cond[-silent_frames_end:, silent_emotion_end] = 1


        sample_copy["output_name"] = create_name(int_idx, emo_idx, identity_idx, training_subjects)

        samples.append(sample_copy)
    return samples


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def interpolate_predictions(first_expression, last_expression, first_jaw_pose, last_jaw_pose, static_frames_start, static_frames_end, num_mouth_closure_frames):
    num_interpolation_frames = num_mouth_closure_frames
    weights = torch.from_numpy( np.linspace(0, 1, num_interpolation_frames)[np.newaxis, ..., np.newaxis]).to(first_expression.device)
    ## add the static frames 
    weights = torch.cat([torch.zeros((1, static_frames_start, 1), dtype=weights.dtype, device=weights.device), weights], dim=1)
    weights = torch.cat([weights, torch.ones((1, static_frames_end, 1), dtype=weights.dtype, device=weights.device)], dim=1)
    interpolated_jaw_pose = last_jaw_pose * weights + first_jaw_pose * (1 - weights)
    interpolated_expression = last_expression * weights.repeat(1,1, 50)  + first_expression * (1 - weights.repeat(1,1, 50))
    return interpolated_expression.float(), interpolated_jaw_pose.float()


def run_evalutation(talking_head, samples, audio_path, overwrite=False, save_meshes=False, pyrender_videos=True, out_folder = None, 
                    silent_start=5, silent_end=5,
                    manual_mouth_closure_start=5, manual_mouth_closure_end=5):
    batch_size = 1
    try:
        template_mesh_path = Path(talking_head.cfg.model.sequence_decoder.flame.flame_lmk_embedding_path).parent / "FLAME_sample.ply" 
    except AttributeError:
        template_mesh_path = Path("/ps/scratch/rdanecek/data/FLAME/geometry/FLAME_sample.ply")
    template = trimesh.load_mesh(template_mesh_path)
    if pyrender_videos:
        renderer = PyRenderMeshSequenceRenderer(template_mesh_path)
    else:
        renderer = None
    D = len(samples)
    BD = int(np.ceil(D / batch_size))
    training_subjects = talking_head.get_subject_labels('training')
    device = talking_head.talking_head_model.device

    # samples = samples[batch_size:]
    dl = torch.utils.data.DataLoader(TestDataset(samples), batch_size=batch_size, shuffle=False, num_workers=0, 
                                     collate_fn=robust_collate)

    # for bd in tqdm(range(BD)):

    #     samples_batch = samples[bd*batch_size:(bd+1)*batch_size]
        # batch = robust_collate(samples_batch)


    if out_folder is None:
        output_dir = Path(talking_head.cfg.inout.full_run_dir) / "test_videos" / (audio_path.parent.name + "_" + audio_path.stem)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = Path(out_folder)
        output_dir.mkdir(exist_ok=True, parents=True)


    for bi, batch in enumerate(tqdm(dl)):
        batch = dict_to_device(batch, device)
        with torch.no_grad():
            batch = talking_head(batch)

        if manual_mouth_closure_start > 0: 
            # first jaw pose 
            last_jaw_pose = batch['predicted_jaw'][:,manual_mouth_closure_start]
            first_jaw_pose = torch.zeros_like(batch['predicted_jaw'][:,0])

            last_expression = batch['predicted_exp'][:,manual_mouth_closure_start] 
            first_expression = torch.zeros_like(batch['predicted_exp'][:,0])

            num_interpolation_frames = manual_mouth_closure_start
            interpolated_expression, interpolated_jaw_pose = interpolate_predictions(first_expression, last_expression, first_jaw_pose, last_jaw_pose, 
                                                                                     static_frames_start=silent_start - manual_mouth_closure_start, 
                                                                                     num_mouth_closure_frames=manual_mouth_closure_start, 
                                                                                     static_frames_end = 0)
            interpolated_expression = torch.zeros_like(interpolated_expression) + last_expression[:, None]

            ## add silence to the audio 
            # silence = torch.zeros((batch["raw_audio"].shape[0], num_interpolation_frames, batch["raw_audio"].shape[2]), dtype=batch["raw_audio"].dtype, device=batch["raw_audio"].device)
            # batch["raw_audio"] = torch.cat([silence, batch["raw_audio"]], dim=1)
            # batch["predicted_jaw"] = torch.cat([interpolated_jaw_pose, batch["predicted_jaw"]], dim=1)
            # batch["predicted_exp"] = torch.cat([interpolated_expression, batch["predicted_exp"]], dim=1)
            batch["predicted_jaw"][:, :interpolated_jaw_pose.shape[1]] = interpolated_jaw_pose
            # batch["predicted_exp"][:, :interpolated_expression.shape[1]]  = interpolated_expression
            
            # pass the interpolated part through FLAME 
            flame = talking_head.talking_head_model.sequence_decoder.get_shape_model()

            pose = torch.cat([torch.zeros_like(interpolated_jaw_pose),interpolated_jaw_pose], dim=-1) 
            exp = batch["predicted_exp"][:, :interpolated_expression.shape[1]]
            B_, T_ = exp.shape[:2]
            exp = exp.view(B_ * T_, -1)
            pose = pose.view(B_ * T_, -1)
            shape = batch["gt_shape"]
            shape = shape[:,None, ...].repeat(1, T_, 1).contiguous().view(B_ * T_, -1)
            predicted_verts, _, _ = flame(shape, exp, pose)
            predicted_verts = predicted_verts.reshape(B_, T_, -1) 
            # batch["predicted_vertices"] = torch.cat([predicted_verts, batch["predicted_vertices"]], dim=1) 
            batch["predicted_vertices"][:, :predicted_verts.shape[1]] = predicted_verts

        
        if manual_mouth_closure_end > 0:
            first_jaw_pose = batch['predicted_jaw'][:,-manual_mouth_closure_end]
            last_jaw_pose = torch.zeros_like(batch['predicted_jaw'][:,-1])

            first_expression = batch['predicted_exp'][:,-manual_mouth_closure_end]
            last_expression = torch.zeros_like(batch['predicted_exp'][:,-1])

            num_interpolation_frames = manual_mouth_closure_end
            interpolated_expression, interpolated_jaw_pose = interpolate_predictions(first_expression, last_expression, first_jaw_pose, last_jaw_pose, 
                                                                                     static_frames_start=0, 
                                                                                     num_mouth_closure_frames=manual_mouth_closure_end, 
                                                                                     static_frames_end = silent_end - manual_mouth_closure_end)
            interpolated_expression = torch.zeros_like(interpolated_expression) + first_expression[:, None]

            ## add silence to the audio
            # silence = torch.zeros((batch["raw_audio"].shape[0], num_interpolation_frames, batch["raw_audio"].shape[2]), dtype=batch["raw_audio"].dtype, device=batch["raw_audio"].device)
            # batch["raw_audio"] = torch.cat([batch["raw_audio"], silence], dim=1)
            # batch["predicted_jaw"] = torch.cat([batch["predicted_jaw"], interpolated_jaw_pose], dim=1)
            # batch["predicted_exp"] = torch.cat([batch["predicted_exp"], interpolated_expression], dim=1)
            batch["predicted_jaw"][:, -interpolated_jaw_pose.shape[1]:] = interpolated_jaw_pose
            # batch["predicted_exp"][:, -interpolated_expression.shape[1]:]  = interpolated_expression

            # pass the interpolated part through FLAME
            flame = talking_head.talking_head_model.sequence_decoder.get_shape_model()

            pose = torch.cat([torch.zeros_like(interpolated_jaw_pose), interpolated_jaw_pose], dim=-1)
            exp = batch["predicted_exp"][:, -interpolated_expression.shape[1]:] 
            B_, T_ = exp.shape[:2]
            exp = exp.view(B_ * T_, -1)
            pose = pose.view(B_ * T_, -1)
            shape = batch["gt_shape"]
            shape = shape[:,None, ...].repeat(1, T_, 1).contiguous().view(B_ * T_, -1)
            predicted_verts, _, _ = flame(shape, exp, pose)
            predicted_verts = predicted_verts.reshape(B_, T_, -1)
            # batch["predicted_vertices"] = torch.cat([batch["predicted_vertices"], predicted_verts], dim=1)
            batch["predicted_vertices"][:, -predicted_verts.shape[1]:] = predicted_verts



        B = batch["predicted_vertices"].shape[0]
        for b in range(B):

            if "output_name" in batch:
                suffix = batch["output_name"][b]
            else:
                try: 
                    intensity = batch["gt_expression_intensity_condition"][b].argmax().item()
                    emotion = batch["gt_expression_label_condition"][b].argmax().item()
                    identity = batch["gt_expression_identity_condition"][b].argmax().item()

                    emotion = AffectNetExpressions(emotion).name
                    identity = training_subjects[identity]
                    suffix = f"_{identity}_{emotion}_{intensity}"
                except Exception as e:
                    print(e)
                    suffix = f"_{bi * batch_size + b}"

            out_audio_path = output_dir / f"{suffix[1:]}" / f"audio.wav"
            import soundfile as sf
            orig_audio, sr = librosa.load(audio_path) 
            ## prepend the silent frames
            if silent_start > 0:
                orig_audio = np.concatenate([np.zeros(int(silent_start * sr / 25), dtype=orig_audio.dtype), orig_audio], axis=0)
            if silent_end > 0:
                orig_audio = np.concatenate([orig_audio, np.zeros(int(silent_end * sr / 25 , ), dtype=orig_audio.dtype)], axis=0)


            if talking_head.render_results:
                predicted_mouth_video = batch["predicted_video"]["front"][b]

                out_video_path = output_dir / f"{suffix[1:]}" / f"pytorch_video.mp4"
                save_video(out_video_path, predicted_mouth_video, fourcc="mp4v", fps=25)
                
                out_audio_path = output_dir / f"{suffix[1:]}" / f"audio.wav"
                sf.write(out_audio_path, orig_audio, samplerate=sr)

                out_video_with_audio_path = output_dir / f"{suffix[1:]}" / f"pytorch_video_with_audio_{suffix}.mp4"

                # attach audio to video with ffmpeg

                ffmpeg_cmd = f"ffmpeg -i {out_video_path} -i {out_audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {out_video_with_audio_path}"
                print(ffmpeg_cmd)
                os.system(ffmpeg_cmd)

                # delete video without audio
                os.remove(out_video_path)
                os.remove(out_audio_path)

            predicted_vertices = batch["predicted_vertices"][b]
            T = predicted_vertices.shape[0]

            out_video_path = output_dir / f"{suffix[1:]}" / f"pyrender.mp4"
            out_video_path.parent.mkdir(exist_ok=True, parents=True)
            out_video_with_audio_path = output_dir / f"{suffix[1:]}" / f"pyrender_with_audio.mp4"

            if save_meshes: 
                mesh_folder = output_dir / f"{suffix[1:]}"  / "meshes"
                mesh_folder.mkdir(exist_ok=True, parents=True)
                for t in tqdm(range(T)):
                    # mesh_path = mesh_folder / (f"{t:05d}" + ".obj")
                    mesh_path = mesh_folder / (f"{t:05d}" + ".ply")
                    if mesh_path.exists() and not overwrite:
                        continue

                    pred_vertices = predicted_vertices[t].detach().cpu().view(-1,3).numpy()
                    mesh = trimesh.base.Trimesh(pred_vertices, template.faces)
                    mesh.export(mesh_path)

                audio_link_path = output_dir / f"{suffix[1:]}" / "audio.wav"
                if not audio_link_path.exists():
                    sf.write(audio_link_path, orig_audio, samplerate=sr)
                    # os.symlink(audio_path, audio_link_path)

            if pyrender_videos:
                if out_video_with_audio_path.exists() and not overwrite:
                    continue

                pred_images = []
                for t in tqdm(range(T)):
                    pred_vertices = predicted_vertices[t].detach().cpu().view(-1,3).numpy()
                    pred_image = renderer.render(pred_vertices)
                    pred_images.append(pred_image)
                    # if save_meshes: 
                    #     mesh = trimesh.base.Trimesh(pred_vertices, template.faces)
                    #     # mesh_path = output_video_dir / (f"frame_{t:05d}" + ".obj")
                    #     mesh.export(mesh_path)

                pred_images = np.stack(pred_images, axis=0)

                save_video(out_video_path, pred_images, fourcc="mp4v", fps=25)


                # sf.write(out_audio_path, batch["raw_audio"][b].view(-1).detach().cpu().numpy(), samplerate=16000)
                sf.write(out_audio_path,  orig_audio, samplerate=sr)

                ffmpeg_cmd = f"ffmpeg -y -i {out_video_path} -i {out_audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {out_video_with_audio_path}"
                print(ffmpeg_cmd)
                os.system(ffmpeg_cmd)

                # delete video without audio
                os.remove(out_video_path)
                os.remove(out_audio_path)

            chmod_cmd = f"find {str(output_dir)} -print -type d -exec chmod 775 {{}} +"
            os.system(chmod_cmd)


def read_audio(audio_path):
    sampling_rate = 16000
    # try:
    wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    # except ValueError: 
    #     import soundfile as sf
    #     wavdata, sampling_rate = sf.read(audio_path, channels=1, samplerate=16000,dtype=np.float32, subtype='PCM_32',format="RAW",endian='LITTLE')
    # wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    if wavdata.ndim > 1:
        wavdata = librosa.to_mono(wavdata)
    wavdata = (wavdata.astype(np.float64) * 32768.0).astype(np.int16)
    # if longer than 30s cut it
    if wavdata.shape[0] > 22 * sampling_rate:
        wavdata = wavdata[:22 * sampling_rate]
        print("Audio longer than 30s, cutting it to 30s")
    return wavdata, sampling_rate


def process_audio(wavdata, sampling_rate, video_fps):
    assert sampling_rate % video_fps == 0 
    wav_per_frame = sampling_rate // video_fps 

    num_frames = wavdata.shape[0] // wav_per_frame

    wavdata_ = np.zeros((num_frames, wav_per_frame), dtype=wavdata.dtype) 
    wavdata_ = wavdata_.reshape(-1)
    if wavdata.size > wavdata_.size:
        wavdata_[...] = wavdata[:wavdata_.size]
    else: 
        wavdata_[:wavdata.size] = wavdata
    wavdata_ = wavdata_.reshape((num_frames, wav_per_frame))
    # wavdata_ = wavdata_[start_frame:(start_frame + num_read_frames)] 
    # if wavdata_.shape[0] < sequence_length:
    #     # concatente with zeros
    #     wavdata_ = np.concatenate([wavdata_, 
    #         np.zeros((sequence_length - wavdata_.shape[0], wavdata_.shape[1]),
    #         dtype=wavdata_.dtype)], axis=0)
    # wavdata_ = wavdata_.astype(np.float64) / np.int16(np.iinfo(np.int16).max)

    # wavdata_ = np.zeros((sequence_length, samplerate // video_fps), dtype=wavdata.dtype)
    # wavdata_ = np.zeros((n * frames.shape[0]), dtype=wavdata.dtype)
    # wavdata_[:wavdata.shape[0]] = wavdata 
    # wavdata_ = wavdata_.reshape((frames.shape[0], -1))
    sample = {}
    sample["raw_audio"] = wavdata_ 
    sample["samplerate"] = sampling_rate
    return sample


## note to self: 
# the MEAD test set that uses the split of "random_by_identityV2_sorted_70_15_15" (all of our training models)
# has the following test individuals: 
# ['M037', 'M039', 'M040', 'M041', 'M042', 'W037', 'W038', 'W040']
# validation individuals: 
# ['M032', 'M033', 'M034', 'M035', 'W033', 'W035', 'W036'] 
# and training individuals
#['M003', # GT status: good emotions, neutral a bit of artifacts, very expressive ## decision: 5
# 'M005',  # GT status: decent emotions, neutral does not look neutral looks not neutral some artifacts, very expressive ## decision: 3.5
# 'M007',  # GT status: decent emotions, neutral some artifacts, ok geometry but weird speaking style ## decision: 3.5
# 'M009',  # GT status: good emotions, neutral a bit of artifacts, very expressive ## decision: 5
# 'M011',  # GT status: ok emotions, neutral does not look neutral at all (bad recostruction) a bit of artifacts, very expressive ## decision: 2.5
# 'M012',  # GT status: ok emotions, neutral a bit of artifacts, quite expressive ## decision: 4.5
# 'M013',  # GT status: black guy, artifacts in reconstruction, quite expressive ## decision: 1
# 'M019',  # GT status: good, emotions, very expressive, tiny bit of neutral artifacts ## decision: 5
# 'M022',  # GT status: good, emotions, very expressive, tiny bit of neutral artifacts ## decision: 5
# 'M023',  # GT status: good, emotions, lower amplitude but good, tiny bit of neutral artifacts ## decision: 5
# 'M024',  # GT status: ok emotions, lower amplitude, neutral mubmles a bit, tiny bit of neutral artifacts ## decision: 4
# 'M025',  # GT status: neural has some artifacts (the black kind of artifacts :-( ) but the emotions are OK: ## decision: 2.5
# 'M026',  # GT status, neutral look emotional so not great, even the happy doesn't look good, not too wrong but overall very weird speaking style: 2
# 'M027',  # GT status: good, emotions, very expressive, tiny bit of neutral artifacts ## decision: 5
# 'M028',  # GT status: good, emotions, very expressive, tiny bit of neutral artifacts ## decision: 5
# 'M029',  # GT status: consistent artifcats around lips (the reconstruction does not work so wel onn this guy): decision: 2 
# 'M030',  #  GT status: ok emotions, ok expressive, some of neutral artifacts ## decision: 4
# 'M031',  # GT status: good, emotions, very expressive, tiny bit of neutral artifacts ## decision: 5
# 'W009', # GT status: 4-5
# 'W011', # GT status: 4-5
# 'W014', # GT status: 4-5
# 'W015', # GT status: 4-5
# 'W016', # GT status: 4-5
# 'W018', # neutral not neutral (black artifact) not good 1 
# 'W019', #  terrible actress but the reconstructions are OK, will confuse people though (happy not happy, ...) GT status: 3
# 'W021', # GT status: 4-5
# 'W023', # neutral not neutral (black artifact) not good 2.5 
# 'W024', # neutral not neutral this time it's actually accurate, this woman't neutral looks sad, that said the reconstructions are OK but will confuse people ## 3.5
# 'W025', # neutral not neutral (black artifact) not good 2. 
# 'W026', # neutral not neutral this time it's actually accurate, this woman't neutral looks sad, that said the reconstructions are OK but will confuse people ## 3.5
# 'W028', # 4-5
# 'W029', # neutral not neutral (black artifact) not good 1.5
# ]
##
training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029']
# val_ids = ['M032', 'M033', 'M034', 'M035', 'W033', 'W035', 'W036'] 
# test_ids = ['M037', 'M039', 'M040', 'M041', 'M042', 'W037', 'W038', 'W040']

# label2score = { 
# 'M003': 5,
# 'M005': 3.5,
# 'M007': 3.5,
# 'M009': 5,
# 'M011': 2.5,
# 'M012': 4.5,
# 'M013': 1,
# 'M019': 5,
# 'M022': 5,
# 'M023': 5,
# 'M024': 4,
# 'M025': 2.5,
# 'M026': 2,
# 'M027': 5,
# 'M028': 5,
# 'M029': 2 ,
# 'M030': 4,
# 'M031': 5,
# 'W009': 4,
# 'W011': 4,
# 'W014': 4,
# 'W015': 4,
# 'W016': 4,
# 'W018': 1 ,
# 'W019': 3,
# 'W021': 4,
# 'W023': 2.5 ,
# 'W024': 3.5,
# 'W025': 2. ,
# 'W026': 3.5,
# 'W028': 4-5,
# 'W029': 1.5,
# }

# id2label = zip(list(training_ids, range(len(training_ids))))
# label2id = zip(range(len(training_ids)), list(training_ids))

# # accepted labels >= 4
# accepted_labels2score = { k: v for k, v in label2score.items() if v >= 4 }
# accaoted_label2id = { k: v for k, v in label2id.items() if v in accepted_labels2score.keys() }


def main(): 
    root = "/is/cluster/work/rdanecek/talkinghead/trainings/"
    # resume_folders = []
    # resume_folders += ["2023_05_04_13-04-51_-8462650662499054253_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"]
    # resume_folders += ["2023_05_04_18-22-17_5674910949749447663_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"]

    if len(sys.argv) > 1:
        resume_folder = sys.argv[1]
    else:
        # good model with disentanglement
        # resume_folder = "2023_05_08_20-36-09_8797431074914794141_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"
        # NEW MAIN MODEL 
        resume_folder = "2023_05_18_01-26-32_-6224330163499889169_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"
    if len(sys.argv) > 2:
        audio = Path(sys.argv[2])
    else:
        # audio = Path('/ps/project/EmotionalFacialAnimation/data/lrs3/extracted/test/0Fi83BHQsMA/00002.mp4')
        # audio = Path('/is/cluster/fast/rdanecek/data/lrs3/processed2/audio/trainval/0af00UcTOSc/50001.wav')
        audio = Path('/is/cluster/fast/rdanecek/data/lrs3/processed2/audio/pretrain/0akiEFwtkyA/00031.wav')
        audio = Path('/home/rdanecek/Downloads/fastforward/01_gday.wav')

    model_path = Path(root) / resume_folder  
    talking_head = TalkingHeadWrapper(model_path, render_results=False)
    eval_talking_head_on_audio(talking_head, audio, 
                               silent_frames_start=30, 
                               silent_frames_end=30, 
                                silent_emotion_start=0, 
                                silent_emotion_end=0,
    )
    


if __name__=="__main__": 
    main()

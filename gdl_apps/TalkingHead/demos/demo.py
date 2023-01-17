import os, sys 
from pathlib import Path
import argparse
from gdl.models.talkinghead.FaceFormer import FaceFormer
from gdl_apps.TalkingHead.utils.load import load_model
from gdl.utils.other import get_path_to_assets
import pickle
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor
import cv2 
import tempfile
import pyrender
import trimesh
# import pyvista as pv
from subprocess import call
from tqdm import auto


def main(): 
    # model_name = "2022_09_05_19-09-35_3326505631348969242_FaceFormer_Facef_Awav2vec2_Elinear_predV" # first trainings
    # model_name = "2022_09_05_19-24-53_-1831723491699865120_FaceFormer_Facef_Awav2vec2_Elinear_predV" # first trainings
    # model_name = "2022_09_06_16-09-09_158799827207587146_FaceFormer_Facef_Awav2vec2_Elinear_DFaceFormerDecoder_predV" # not good enough
    # model_name = "2022_09_06_18-57-51_-4531631262283512130_FaceFormer_Facef_Awav2vec2_Elinear_DFaceFormerDecoder_predV" # not good enough
    # model_name = "2022_09_06_18-58-04_-8145049632705930270_FaceFormer_Facef_Awav2vec2_Elinear_DFlameFormerDecoder_predEJ" # not good enough
    # model_name = "2022_09_07_15-23-32_9054186072148635306_FaceFormer_Facef_Awav2vec2_Elinear_DFaceFormerDecoder_predV" # not good enough
    # model_name = "2022_09_08_18-22-35_-5082924037183487904_FaceFormer_Facef_Awav2vec2T_Elinear_DFaceFormerDecoder_predV" # promising
    # model_name = "2022_09_08_18-21-12_-1074603920524203560_FaceFormer_Facef_Awav2vec2T_Elinear_DFaceFormerDecoder_predV" # promising
    # model_name = "2022_09_08_18-22-36_6903650459323198617_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoder_predEJ"
    # model_name = "2022_09_08_18-38-39_2713943321348760067_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoder_predEJ"
    # model_name = "2022_09_09_13-32-26_-1721131782444898775_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoder_predEJ" # long training, 6D prediction
    # model_name = "2022_09_10_20-03-08_-1412790012237496553_FaceFormer_Facef_Awav2vec2T_Elinear_DFaceFormerDecoder_predV_LV" # faceformer, velocity metric not used but measured
    # model_name = "2022_09_10_20-07-21_-3441816657521570286_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoder_predEJ_LVVv" # not yet finished FlameFormer 
    # model_name = "2022_09_10_20-03-27_7326717866113306532_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoder_predEJ_LV"
    # model_name = "2022_09_10_20-07-18_8911584479033387564_FaceFormer_Facef_Awav2vec2T_Elinear_DFaceFormerDecoder_predV_LVVv" # faceformer with velocity
    # model_name = "2022_09_10_20-18-06_-3860295090740618656_FaceFormer_Facef_Awav2vec2T_Elinear_DBertDecoder1_predV_LV" # bert-like
    # model_name = "2022_09_10_20-18-26_-2697804222243016036_FaceFormer_Facef_Awav2vec2T_Elinear_DBertDecoder1_predV_LVVv" # bert-like with velocity
    # model_name = "2022_09_11_15-31-29_556028787792822791_FaceFormer_Facef_Awav2vec2T_Elinear_DFaceFormerDecoder1NPE_predV_LV" # faceformer no positional enc
    # model_name = "2022_09_11_15-27-58_7969814270557703130_FaceFormer_Facef_Awav2vec2T_Elinear_DFaceFormerDecoder4PPE_predV_LV" # faceformer, 4 decoder layers ## GOOD
    # model_name = "2022_09_11_15-28-13_-8699916949864340316_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoder4PPE_predEJ_LVVv" # flameformer 4 layers, velocity
    # model_name = "2022_09_11_15-27-26_-2679574166217615782_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoder4PPE_predEJ_LV" # flameformer 4 layers
    # model_name = "2022_09_12_14-10-39_-8321657434285622693_FaceFormer_Facef_Awav2vec2T_Elinear_DBertDecoder4NPE_NAB_predV_LV" # bert with zero init for last layer, no positional enc, no attention bias
    # model_name = "2022_09_12_14-10-23_-6578255391856750064_FaceFormer_Facef_Awav2vec2T_Elinear_DBertDecoder4PPE_NAB_predV_LV" # bert with zero init for last layer, positional enc, no attention bias
    # model_name = "2022_09_12_14-10-27_697855882805977705_FaceFormer_Facef_Awav2vec2T_Elinear_DBertDecoder1NPE_NAB_predV_LV" # bert with zero init for last layer, no positional enc, no attention bias
    # model_name = "2022_09_12_14-10-00_6383928147122437552_FaceFormer_Facef_Awav2vec2T_Elinear_DBertDecoder1PPE_NAB_predV_LV" # bert with zero init for last layer, positional enc, no attention bias
    # model_name = "2022_09_12_19-10-20_-7648949959771599137_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameBertDecoder4PPE_predEJ_LV" # flamebert
    # model_name = "2022_09_12_19-10-15_1345185436813455300_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameBertDecoder4PPE_Tff_predEJ_LV"
    # model_name = "2022_09_12_15-46-18_3526214713565015856_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoder1PPE_NAB_predEJ_LV"
    # model_name = "2022_09_13_08-44-20_-4903461877341645113_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameBertDecoder1NPE_predEJ_LV"
    # model_name = "2022_09_14_15-02-46_-2516580837500186411_FaceFormer_Facef_Awav2vec2T_Elinear_DMLPDecoderNPE_predV_LV" # mlp
    # model_name = "2022_09_14_15-04-42_-6705703325717568620_FaceFormer_Facef_Awav2vec2T_Elinear_DLinearDecoderNPE_predV_LV" # linear

    # models without conditionng
    # model_name = "2022_09_14_10-01-55_4418334335717242648_FaceFormer_Facef_Awav2vec2T_Elinear_DBertDecoder1_SnoPPE_Tff_predV_LV"
    # model_name = "2022_09_14_10-02-06_7507315077752011123_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameBertDecoder1_SnoNPE_Tff_predEJ_LV"
    # model_name = "2022_09_14_15-01-52_435416857314710552_FaceFormer_Facef_Awav2vec2T_Elinear_DLinearDecoder_SnoNPE_predV_LV" # linear
    # model_name = "2022_09_14_15-01-52_-8313895241432443176_FaceFormer_Facef_Awav2vec2T_Elinear_DMLPDecoder_SnoNPE_predV_LV" # mlp

    # sanity checks after some developing 
    # model_name = "2022_10_13_09-09-57_-979709473838080274_FaceFormer_Facef_Awav2vec2_Elinear_DFlameBertDecoderNPE_predEJ_LV" # flamebert, w2v not trainable, not good
    # model_name = "2022_10_13_10-09-47_7813958024996471243_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoderPPE_predEJ_LV" # flameformer, 4 layers # not yet finished running
    model_name = "2022_10_13_09-07-22_1102753037878748995_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameFormerDecoderPPE_predEJ_LV" # flameformer, # not yet finished running
    # model_name = "2022_10_13_10-10-16_7461908104531570234_FaceFormer_Facef_Awav2vec2T_Elinear_DFaceFormerDecoderPPE_predV_LV" # faceformer # not yet finished but seems to work
    # model_name = "2022_10_13_10-10-16_-3320561324617337278_FaceFormer_Facef_Awav2vec2T_Elinear_DFlameBertDecoderNPE_predEJ_LV" # finished, works, no position enc

    # add argparser here
    parser = argparse.ArgumentParser(description='Talking Head') 
    parser.add_argument('--path_to_models', type=str, default='/is/cluster/work/rdanecek/talkinghead/trainings', help='Path to trained models')
    parser.add_argument('--model_name', type=str, default=model_name, help='Name of the model to use')
    parser.add_argument('--mode', type=str, default='latest', help='Checkpoint to use (best vs latest)')
    parser.add_argument('--wav_path', type=str, 
        default='/is/cluster/fast/rdanecek/data/lrs3/processed2/audio/pretrain/0af00UcTOSc/00021.wav', 
        help='Checkpoint to use (best vs latest)')
    # parser.add_argument('--condition', type=int, default=0, help='Index of the conditioning training subject')
    parser.add_argument('--condition', type=str, default='FaceTalk_170913_03279_TA', help='Index of the conditioning training subject')
    # parser.add_argument('--subject', type=int, default=0, help='Index of the template training subject to animate')
    parser.add_argument('--subject', type=str, default='FaceTalk_170809_00138_TA', help='Index of the template training subject to animate')
    parser.add_argument('--fps', type=int, default=30, help='FPS of the video')
    parser.add_argument('--save_meshes', action='store_true', help='Saves the meshes as .ply files')
    # parser.add_argument('--save_meshes', default=True, action='store_true', help='Saves the meshes as .ply files')
    # parser.add_argument('--result_path', type=str, default='output', help='Path to save the output')

    args = parser.parse_args()

    model, cfg = load_model(args.path_to_models, args.model_name, args.mode)
    model.eval()
    model.cuda()

    # template_file = os.path.join(args.dataset, args.template_path)
    template_file = cfg.data.template_file
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    train_subjects_list = [i for i in cfg.data.train_subjects]

    one_hot_labels = np.eye(len(train_subjects_list))
    if isinstance(args.condition, int):
        iter = args.condition
    elif isinstance(args.condition, str):
        iter = train_subjects_list.index(args.condition)
    else:
        raise ValueError("condition must be either int or str")
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=model.device)

    if isinstance(args.subject, int):
        subject = train_subjects_list[args.subject]
    elif isinstance(args.subject, str):
        subject = args.subject
    else:
        raise ValueError("condition must be either int or str")
    temp = templates[subject]
             
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device=model.device)

    wav_path = args.wav_path
    test_name = Path(wav_path).stem
    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
    processor = Wav2Vec2Processor.from_pretrained(cfg.model.audio.model_specifier)
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=model.device)

    batch = {} 
    batch['processed_audio'] = audio_feature
    batch['template'] = template#.unsqueeze(0)
    batch['one_hot'] = one_hot#.unsqueeze(0)

    with torch.no_grad():
        prediction = model(batch, train=False)
    prediction = prediction['predicted_vertices'].squeeze().detach().cpu().numpy() # (seq_len, V*3)
    # out_path = Path(args.result_path) / test_name
    out_path = Path(args.path_to_models) / args.model_name / "test_results" / test_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    render_sequence(args, cfg, subject, train_subjects_list[iter], prediction, out_path)


def render_sequence(args, cfg, subject, condition, predicted_vertices, out_path):
    wav_path = Path(args.wav_path)
    test_name = wav_path.stem
    # predicted_vertices_path = os.path.join(args.result_path,test_name+".npy")
    # if args.dataset == "BIWI":
    #     template_file = os.path.join(args.dataset, args.render_template_path, "BIWI.ply")
    # elif args.dataset == "vocaset":
    #     template_file = os.path.join(args.dataset, args.render_template_path, "FLAME_sample.ply")
    # template_file = os.path.join(args.render_template_path, "FLAME_sample.ply") 
    template_file = get_path_to_assets() / "FLAME" / "geometry" / "FLAME_sample.ply"
        
    print("rendering: ", test_name)
                 
    # template = Mesh(filename=template_file)
    template = trimesh.load_mesh(template_file)
    # predicted_vertices = np.load(predicted_vertices_path)
    predicted_vertices = np.reshape(predicted_vertices,(-1, *template.vertices.shape))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)
    center = np.mean(predicted_vertices[0], axis=0)

    for i_frame in auto.tqdm(range(num_frames)):
        # render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        render_mesh = trimesh.base.Trimesh(predicted_vertices[i_frame], template.faces)
        pred_img = render_mesh_helper(args, cfg, render_mesh, center)
        pred_img = pred_img.astype(np.uint8)
        # save mesh 
        if args.save_meshes:
            # mesh = Mesh(predicted_vertices[i_frame], template.f)
            render_mesh.export(os.path.join(out_path, 'mesh_{:04d}.ply'.format(i_frame)))
        writer.write(pred_img)

    writer.release()
    file_name = test_name + "_" + subject + "_condition_" + condition

    np.save( os.path.join(out_path, file_name + '.pkl'), predicted_vertices)

    video_fname_tmp = os.path.join(out_path, file_name + '_.mp4')
    # cmd = f'ffmpeg -y -i {tmp_video_file.name} -i {str(wav_path)} -pix_fmt yuv420p -qscale 0 -map 1:a -shortest {video_fname}'
    cmd = f'ffmpeg -y -i {tmp_video_file.name} -pix_fmt yuv420p -qscale 0 {video_fname_tmp}'
    # call(cmd.split(' '))
    os.system(cmd)
    video_fname = os.path.join(out_path, file_name + '.mp4')
    cmd = f'ffmpeg -y -i {video_fname_tmp} -i {str(wav_path)}  -c copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k {video_fname}'
    os.system(cmd)
    os.remove(video_fname_tmp)




def render_mesh_helper(args, cfg , mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    # if args.dataset == "BIWI":
    #     camera_params = {'c': np.array([400, 400]),
    #                      'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
    #                      'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    # elif args.dataset == "vocaset":
    camera_params = {'c': np.array([400, 400]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    # mesh_copy =  Mesh(mesh.v, mesh.f)
    transformed_verts = mesh.vertices
    transformed_verts = cv2.Rodrigues(rot)[0].dot((transformed_verts-t_center).T).T+t_center
    mesh_copy = trimesh.base.Trimesh(transformed_verts, mesh.faces)
    # mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

    # tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(mesh_copy, material=primitive_material,smooth=True)

    background_black = True
    if background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]



if __name__ == "__main__": 
    main()

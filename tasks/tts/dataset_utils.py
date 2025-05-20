import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0, f0_to_coarse
from utils.commons.dataset_utils import (
    BaseDataset,
    collate_1d_or_2d,
    collate_1d,
    collate_2d,
)
from utils.commons.indexed_datasets import IndexedDataset
from utils.text.text_encoder import build_token_encoder
from utils.text import intersperse
import os
import torch.nn.functional as F
import random
import utils
import librosa
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize as librosa_normalize
from tasks.tts.augmentation import Augment
import torchaudio

def load_filename_pairs(filepath):
    filename_pairs = {}
    with open(filepath, 'r') as file:
        for line in file:
            original, new = line.strip().split(',')
            filename_pairs[original.strip()] = new.strip()
    return filename_pairs

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        y = torch.clamp(y, -1.0, 1.0)
    if torch.max(y) > 1.0:
        y = torch.clamp(y, -1.0, 1.0)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    
    spec = torch.view_as_real(spec)
    linear = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], linear)
    spec = spectral_normalize_torch(spec)
    return spec

class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams

        self.data_dir = hparams["binary_data_dir"] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == "test" and len(hparams["test_ids"]) > 0:
                self.avail_idxs = hparams["test_ids"]
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == "train" and hparams["min_frames"] > 0:
                self.avail_idxs = [
                    x for x in self.avail_idxs if self.sizes[x] >= hparams["min_frames"]
                ]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, "avail_idxs") and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item["mel"]) == self.sizes[index], (
            len(item["mel"]),
            self.sizes[index],
        )
        max_frames = hparams["max_frames"]
        spec = torch.Tensor(item["mel"])[:max_frames]
        max_frames = (
            spec.shape[0] // hparams["frames_multiple"] * hparams["frames_multiple"]
        )
        spec = spec[:max_frames]
        ph_token = torch.LongTensor(item["ph_token"][: hparams["max_input_tokens"]])
        sample = {
            "id": index,
            "item_name": item["item_name"],
            "text": item["txt"],
            "txt_token": ph_token,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams["use_spk_embed"]:
            sample["spk_embed"] = torch.Tensor(item["spk_embed"])
        if hparams["use_spk_id"]:
            sample["spk_id"] = int(item["spk_id"])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s["id"] for s in samples])
        item_names = [s["item_name"] for s in samples]
        text = [s["text"] for s in samples]
        txt_tokens = collate_1d_or_2d([s["txt_token"] for s in samples], 0)
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s["txt_token"].numel() for s in samples])
        mel_lengths = torch.LongTensor([s["mel"].shape[0] for s in samples])
        batch = {
            "id": id,
            "item_name": item_names,
            "nsamples": len(samples),
            "text": text,
            "txt_tokens": txt_tokens,
            "txt_lengths": txt_lengths,
            "mels": mels,
            "mel_lengths": mel_lengths,
        }

        if hparams["use_spk_embed"]:
            spk_embed = torch.stack([s["spk_embed"] for s in samples])
            batch["spk_embed"] = spk_embed
        if hparams["use_spk_id"]:
            spk_ids = torch.LongTensor([s["spk_id"] for s in samples])
            batch["spk_ids"] = spk_ids
        return batch


class FastSpeechDataset(BaseSpeechDataset):
    def __getitem__(self, index):
        sample = super(FastSpeechDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample["mel"]
        T = mel.shape[0]
        ph_token = sample["txt_token"]
        sample["mel2ph"] = mel2ph = torch.LongTensor(item["mel2ph"])[:T]
        if hparams["use_pitch_embed"]:
            assert "f0" in item
            pitch = torch.LongTensor(item.get(hparams.get("pitch_key", "pitch")))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if hparams["pitch_type"] == "ph":
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item["f0_ph"])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = (
                    torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                )
                f0_phlevel_num = (
                    torch.zeros_like(ph_token)
                    .float()
                    .scatter_add(0, mel2ph - 1, torch.ones_like(f0))
                    .clamp_min(1)
                )
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(FastSpeechDataset, self).collater(samples)
        hparams = self.hparams
        if hparams["use_pitch_embed"]:
            f0 = collate_1d_or_2d([s["f0"] for s in samples], 0.0)
            pitch = collate_1d_or_2d([s["pitch"] for s in samples])
            uv = collate_1d_or_2d([s["uv"] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        mel2ph = collate_1d_or_2d([s["mel2ph"] for s in samples], 0.0)
        batch.update(
            {
                "mel2ph": mel2ph,
                "pitch": pitch,
                "f0": f0,
                "uv": uv,
            }
        )
        return batch


class GradTTSDataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        data_dir = self.hparams["processed_data_dir"]
        self.token_encoder = build_token_encoder(f"{data_dir}/phone_set.json")

    def __getitem__(self, index):
        item = self._get_item(index)    
        sample = super().__getitem__(index)
        ph_token = sample["txt_token"]
        ph_token = intersperse(ph_token, len(self.token_encoder))
        ph_token = torch.IntTensor(ph_token)
        sample["txt_token"] = ph_token
        sample["emo_id"] = int(item['emo_id'])
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        emo_ids = torch.LongTensor([s['emo_id'] for s in samples])
        batch.update({"emo_ids": emo_ids})
        return batch

class FastSpeech2Dataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        self.pitch_type = self.hparams.get("pitch_type")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample["mel"]
        T = mel.shape[0]
        sample["energy"] = (mel.exp() ** 2).sum(-1).sqrt()
        cwt_spec = torch.Tensor(item["cwt_spec"])[:T]
        f0_mean = item.get("f0_mean", item.get("cwt_mean"))
        f0_std = item.get("f0_std", item.get("cwt_std"))
        sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        energy = collate_1d([s["energy"] for s in samples], 0.0)
        batch.update({"energy": energy})
        cwt_spec = collate_2d([s["cwt_spec"] for s in samples])
        f0_mean = torch.Tensor([s["f0_mean"] for s in samples])
        f0_std = torch.Tensor([s["f0_std"] for s in samples])
        batch.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        return batch


class Tacotron2Dataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super(Tacotron2Dataset, self).__init__(prefix, shuffle, items, data_dir)
        
    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)    
        sample = super(Tacotron2Dataset, self).__getitem__(index)
        mel = sample["mel"]
        T = mel.shape[0]
        gate = torch.zeros(T)
        gate[-1] = 1
        sample["gate"] = gate
        return sample

    def collater(self, samples):
        hparams = self.hparams
        batch = super(Tacotron2Dataset, self).collater(samples)
        gate = collate_1d_or_2d([s["gate"] for s in samples], 1.0)
        batch.update({"gates": gate})
        return batch


class TransformerTTSDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super(TransformerTTSDataset, self).__init__(prefix, shuffle, items, data_dir)

    def __getitem__(self, index):
        sample = super(TransformerTTSDataset, self).__getitem__(index)
        mel = sample["mel"]
        T = mel.shape[0]
        gate = torch.zeros(T)
        gate[-1] = 1
        sample["gate"] = gate
        text_length = sample["txt_token"].numel()
        pos_text = torch.arange(1, text_length + 1)
        pos_mel = torch.arange(1, T + 1)
        sample["pos_text"] = pos_text
        sample["pos_mel"] = pos_mel
        return sample

    def collater(self, samples):
        batch = super(TransformerTTSDataset, self).collater(samples)
        gate = collate_1d_or_2d([s["gate"] for s in samples], 1.0)
        pos_text = collate_1d_or_2d([s["pos_text"] for s in samples], 1.0)
        pos_mel = collate_1d_or_2d([s["pos_mel"] for s in samples], 1.0)

        batch.update({"gates": gate})
        batch.update({"pos_text": pos_text})
        batch.update({"pos_mel": pos_mel})
        return batch

class DINO_5crop_FS_cluster(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        self.pitch_type = self.hparams.get("pitch_type")
        self.emo_wav_dict = self._parse_txt_file('/workspace/none/sd0/DiEmo_final/list/DINO_ESD_train_cluster_final.txt')
        self.Augment = Augment()

    def _parse_txt_file(self, txt_file):
        emo_wav_dict = {}
        with open(txt_file, 'r') as f:
            for line in f:
                wav_name, emo_id = line.strip().split(', ')
                emo_id = int(emo_id)
                if emo_id not in emo_wav_dict:
                    emo_wav_dict[emo_id] = []
                emo_wav_dict[emo_id].append(wav_name)
        return emo_wav_dict

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        sample["emo_id"] = emo_id = int(item['emo_id'])
        
        # Crop
        other_mels = [wav for wav in self.emo_wav_dict[emo_id] if wav != item['item_name']]
        if len(other_mels) >= 2:
            ALL_mels = random.sample(other_mels, 4)
        ALL_mels.append(item['item_name'])
        
        wav_path = "/workspace/none/hd0/dataset/ESD_DINO_mel/wavs"
        #####################################
        base_wav_path = item['item_name'] + ".pt"
        base_wav_full_path = os.path.join(wav_path, base_wav_path)
        base_selected_wav = torch.load(base_wav_full_path).unsqueeze(0).detach() # [L, 80]
        
        base_selected_mel = mel_spectrogram(
            base_selected_wav, 
            n_fft=hparams['fft_size'],
            num_mels=hparams['audio_num_mel_bins'],
            sampling_rate=hparams['audio_sample_rate'],
            hop_size=hparams['hop_size'],
            win_size=hparams['win_size'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            center=False,
            )
        base_selected_mel = base_selected_mel.squeeze(0).T.numpy()
        
        max_frames = hparams["max_frames"]
        mel_gen_spec = torch.Tensor(base_selected_mel)[:max_frames]
        max_frames = (
            mel_gen_spec.shape[0] // hparams["frames_multiple"] * hparams["frames_multiple"]
        )
        mel_gen_spec = mel_gen_spec[:max_frames]

        sample["mel"] = mel_gen_spec
        
        mel = sample["mel"]
        T = mel.shape[0]
        sample["energy"] = (mel.exp() ** 2).sum(-1).sqrt()
        cwt_spec = torch.Tensor(item["cwt_spec"])[:T]
        f0_mean = item.get("f0_mean", item.get("cwt_mean"))
        f0_std = item.get("f0_std", item.get("cwt_std"))
        sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        
        global_l = 187
        local_l = 125
        
        for i in range(6):
            selected_wav_path = random.choice(ALL_mels) + ".pt"
            wav_full_path = os.path.join(wav_path, selected_wav_path)
            selected_wav = torch.load(wav_full_path).unsqueeze(0).detach() # [L, 80]
            aug_selected_wav = self.Augment(selected_wav)
            
            selected_mel = mel_spectrogram(
                aug_selected_wav, 
                n_fft=hparams['fft_size'],
                num_mels=hparams['audio_num_mel_bins'],
                sampling_rate=hparams['audio_sample_rate'],
                hop_size=hparams['hop_size'],
                win_size=hparams['win_size'],
                fmin=hparams['fmin'],
                fmax=hparams['fmax'],
                center=False,
                )
            selected_mel = selected_mel.squeeze(0).transpose(0,1)
            
            selected_mel_l = selected_mel.shape[0]
            mel_l = mel.shape[0]
                        
            if i < 2:
                if selected_mel_l >= global_l:
                    ran_num = random.randint(0, selected_mel_l-global_l)
                    global_crop = selected_mel[ran_num:ran_num+global_l , :]
                else:
                    global_crop = selected_mel
                
                crop_name = f"global_crop_{i}"
                sample[crop_name] = global_crop    
            
            elif i == 2:
                if mel_l >= local_l:
                    ran_num = random.randint(0, mel_l-local_l)
                    local_crop = mel[ran_num:ran_num+local_l , :]
                else:
                    local_crop = mel
                    
                crop_name = f"local_crop_{i}"
                sample[crop_name] = local_crop
                
            else:
                if selected_mel_l >= local_l:
                    ran_num = random.randint(0, selected_mel_l-local_l)
                    local_crop = selected_mel[ran_num:ran_num+local_l , :]
                else:
                    local_crop = selected_mel
                
                crop_name = f"local_crop_{i}"
                sample[crop_name] = local_crop
                    
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        energy = collate_1d([s["energy"] for s in samples], 0.0)
        batch.update({"energy": energy})
        cwt_spec = collate_2d([s["cwt_spec"] for s in samples])
        f0_mean = torch.Tensor([s["f0_mean"] for s in samples])
        f0_std = torch.Tensor([s["f0_std"] for s in samples])
        emo_ids = torch.LongTensor([s['emo_id'] for s in samples])
        
        global_crop_0 = collate_1d_or_2d([s["global_crop_0"] for s in samples], 0.0)
        global_crop_1 = collate_1d_or_2d([s["global_crop_1"] for s in samples], 0.0)
        local_crop_2 = collate_1d_or_2d([s["local_crop_2"] for s in samples], 0.0)
        local_crop_3 = collate_1d_or_2d([s["local_crop_3"] for s in samples], 0.0)
        local_crop_4 = collate_1d_or_2d([s["local_crop_4"] for s in samples], 0.0)
        local_crop_5 = collate_1d_or_2d([s["local_crop_5"] for s in samples], 0.0)
        
        global_crop_0_lengths = torch.LongTensor([s["global_crop_0"].shape[0] for s in samples])
        global_crop_1_lengths = torch.LongTensor([s["global_crop_1"].shape[0] for s in samples])
        local_crop_2_lengths = torch.LongTensor([s["local_crop_2"].shape[0] for s in samples])
        local_crop_3_lengths = torch.LongTensor([s["local_crop_3"].shape[0] for s in samples])
        local_crop_4_lengths = torch.LongTensor([s["local_crop_4"].shape[0] for s in samples])
        local_crop_5_lengths = torch.LongTensor([s["local_crop_5"].shape[0] for s in samples])
        
        batch.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std, "emo_ids": emo_ids, "global_crop_0": global_crop_0, "global_crop_1": global_crop_1, "local_crop_2": local_crop_2, "local_crop_3": local_crop_3, "local_crop_4": local_crop_4, "local_crop_5": local_crop_5, "global_crop_0_lengths": global_crop_0_lengths, "global_crop_1_lengths": global_crop_1_lengths, "local_crop_2_lengths": local_crop_2_lengths, "local_crop_3_lengths": local_crop_3_lengths, "local_crop_4_lengths": local_crop_4_lengths, "local_crop_5_lengths": local_crop_5_lengths})
        return batch


class DINO_infer(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        self.pitch_type = self.hparams.get("pitch_type")
        self.filename_pairs = load_filename_pairs('/workspace/none/sd0/DiEmo_final/list/test_pair_wav.txt')
        
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        
        mel = sample["mel"]
        T = mel.shape[0]
        sample["energy"] = (mel.exp() ** 2).sum(-1).sqrt()
        cwt_spec = torch.Tensor(item["cwt_spec"])[:T]
        f0_mean = item.get("f0_mean", item.get("cwt_mean"))
        f0_std = item.get("f0_std", item.get("cwt_std"))
        sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        sample["emo_id"] = int(item['emo_id'])
        
        item_name= item["item_name"]
        file_name = self.filename_pairs[item_name] + ".pt"
        
        wav_path = "/workspace/none/hd0/dataset/ESD_DINO_mel/wavs"
        base_wav_full_path = os.path.join(wav_path, file_name)
        base_selected_wav = torch.load(base_wav_full_path).unsqueeze(0).detach() # [L, 80]
        
        base_selected_mel = mel_spectrogram(
            base_selected_wav, 
            n_fft=hparams['fft_size'],
            num_mels=hparams['audio_num_mel_bins'],
            sampling_rate=hparams['audio_sample_rate'],
            hop_size=hparams['hop_size'],
            win_size=hparams['win_size'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            center=False,
            )
        base_selected_mel = base_selected_mel.squeeze(0).T.numpy()
        
        max_frames = hparams["max_frames"]
        mel_gen_spec = torch.Tensor(base_selected_mel)[:max_frames]
        max_frames = (
            mel_gen_spec.shape[0] // hparams["frames_multiple"] * hparams["frames_multiple"]
        )
        mel_gen_spec = mel_gen_spec[:max_frames]
        sample["mel"] = mel_gen_spec.detach()
       
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        energy = collate_1d([s["energy"] for s in samples], 0.0)
        batch.update({"energy": energy})
        cwt_spec = collate_2d([s["cwt_spec"] for s in samples])
        f0_mean = torch.Tensor([s["f0_mean"] for s in samples])
        f0_std = torch.Tensor([s["f0_std"] for s in samples])
        emo_ids = torch.LongTensor([s['emo_id'] for s in samples])
        batch.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std, "emo_ids": emo_ids})
        return batch
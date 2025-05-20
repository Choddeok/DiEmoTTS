import torch
from torch import nn
from models.commons.layers import Embedding
from models.commons.nar_tts_modules import EnergyPredictor, PitchPredictor
from models.tts.commons.align_ops import expand_states
from models.tts.all_fastspeech_DCT import FastSpeech
from utils.audio.cwt import cwt2f0, get_lf0_cwt
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse, norm_f0
import numpy as np

from models.tts.DiEmoTTS.encoder_module import MelStyleEncoder, DINOHead
from models.tts.DiEmoTTS.momentum import cosine_scheduler
import math

class ExpressiveFS2(FastSpeech):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)
        self.pitch_embed = Embedding(300, self.hidden_size, 0)
        self.energy_embed = Embedding(300, self.hidden_size, 0)
        self.energy_predictor = EnergyPredictor(
            self.hidden_size,
            n_chans=self.hidden_size,
            n_layers=hparams["predictor_layers"],
            dropout_rate=hparams["predictor_dropout"],
            odim=2,
            kernel_size=hparams["predictor_kernel"],
        )
        self.pitch_predictor = PitchPredictor(
            self.hidden_size,
            n_chans=self.hidden_size,
            n_layers=hparams["predictor_layers"],
            dropout_rate=hparams["predictor_dropout"],
            odim=11,
            kernel_size=hparams["predictor_kernel"],
        )
        self.cwt_stats_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
        )
        self.spk_id_proj = Embedding(hparams["num_spk"], self.hidden_size)
        self.student_encoder = MelStyleEncoder(self.hidden_size)
        self.teacher_encoder = MelStyleEncoder(self.hidden_size)
        self.student_head = DINOHead(in_dim = 256, 
                 out_dim=hparams["num_emo"], 
                 use_bn=hparams["use_bn_in_head"], norm_last_layer=hparams["norm_last_layer"])
        self.teacher_head = DINOHead(in_dim = 256, 
                 out_dim=hparams["num_emo"], 
                 use_bn=hparams["use_bn_in_head"])
        self.nepoch = math.ceil((160000 * hparams["max_sentences"]) / 12155)
        self.momentum_schedule = cosine_scheduler(0.996, 1, self.nepoch, 12155)
        
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        
    def forward(
        self,
        txt_tokens,
        mel2ph=None,
        target=None,
        spk_id=None,
        emo_id=None,
        f0=None,
        uv=None,
        energy=None,
        infer=False,
        **kwargs
    ):
        ret = {}
        # emo
        emo_embed = 0
        emo_embed = emo_embed + self.student_encoder(target)
        emo_embed = emo_embed[:, None, :]
        
        # spk
        spk_embed = 0
        spk_embed = spk_embed + self.spk_id_proj(spk_id)
        spk_embed = spk_embed[:, None, :]
        
        encoder_out = self.encoder(txt_tokens, emo_cond=emo_embed, spk_cond=spk_embed)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        
        # add dur
        dur_inp = (encoder_out) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret, infer)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = decoder_inp_ = expand_states(encoder_out, mel2ph)

        # add pitch and energy embed
        pitch_inp = (decoder_inp_) * tgt_nonpadding
        decoder_inp = decoder_inp + self.forward_pitch(
            pitch_inp, f0, uv, mel2ph, ret, infer
        )

        # add pitch and energy embed
        energy_inp = (decoder_inp_) * tgt_nonpadding
        decoder_inp = decoder_inp + self.forward_energy(energy_inp, energy, ret, infer)

        # decoder input
        decoder_inp = decoder_inp
        
        ret["mel_out"] = self.forward_decoder(
            decoder_inp, tgt_nonpadding, emo_cond=emo_embed, spk_cond=spk_embed,ret=ret, infer=infer, **kwargs
        )
        return ret
        
    def compute_loss(
        self,
        txt_tokens,
        mel2ph=None,
        target=None,
        global_crop_0=None,
        global_crop_1=None,
        local_crop_2=None,
        local_crop_3=None,
        local_crop_4=None,
        local_crop_5=None,
        global_crop_0_lengths=None,
        global_crop_1_lengths=None,
        local_crop_2_lengths=None,
        local_crop_3_lengths=None,
        local_crop_4_lengths=None,
        local_crop_5_lengths=None,
        spk_id=None,
        emo_id=None,
        f0=None,
        uv=None,
        energy=None,
        it=None,
        infer=False,
        **kwargs
    ):
        ret = {}
        
        # spk
        spk_embed = 0
        spk_embed = spk_embed + self.spk_id_proj(spk_id)
        
        # emo
        # global_crop
        global_emb_0 = self.teacher_encoder(global_crop_0, global_crop_0_lengths)
        global_head_0 = self.teacher_head(global_emb_0)
        
        global_emb_1 = self.teacher_encoder(global_crop_1, global_crop_1_lengths)
        global_head_1 = self.teacher_head(global_emb_1)

        ret["teacher_emb"] = torch.cat((global_emb_0, global_emb_1), dim=0) 
        ret["teacher_head"] = torch.cat((global_head_0, global_head_1), dim=0)
         
        # local_crop
        stu_emb_0 = self.student_encoder(global_crop_0, global_crop_0_lengths)
        stu_head_0 = self.student_head(stu_emb_0)
        
        stu_emb_1 = self.student_encoder(global_crop_1, global_crop_1_lengths)
        stu_head_1 = self.student_head(stu_emb_1)
        
        stu_emb_2 = self.student_encoder(local_crop_2, local_crop_2_lengths)
        stu_head_2 = self.student_head(stu_emb_2)
                
        stu_emb_3 = self.student_encoder(local_crop_3, local_crop_3_lengths)
        stu_head_3 = self.student_head(stu_emb_3)
                
        stu_emb_4 = self.student_encoder(local_crop_4, local_crop_4_lengths)
        stu_head_4 = self.student_head(stu_emb_4)
                
        stu_emb_5 = self.student_encoder(local_crop_5, local_crop_5_lengths)
        stu_head_5 = self.student_head(stu_emb_5)

        ret["student_emb"] = torch.cat((stu_emb_0, stu_emb_1, stu_emb_2, stu_emb_3, stu_emb_4, stu_emb_5), dim=0) 
        ret["student_head"] = torch.cat((stu_head_0, stu_head_1, stu_head_2, stu_head_3, stu_head_4, stu_head_5), dim=0) 
        
        # ALl
        stu_emb_2 = stu_emb_2[:, None, :]
        spk_embed = spk_embed[:, None, :]
        
        encoder_out = self.encoder(txt_tokens, emo_cond=stu_emb_2, spk_cond=spk_embed)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        
        # add dur
        dur_inp = (encoder_out) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret, infer)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = decoder_inp_ = expand_states(encoder_out, mel2ph)

        # add pitch and energy embed
        pitch_inp = (decoder_inp_) * tgt_nonpadding
        decoder_inp = decoder_inp + self.forward_pitch(
            pitch_inp, f0, uv, mel2ph, ret, infer
        )

        # add pitch and energy embed
        energy_inp = (decoder_inp_) * tgt_nonpadding
        decoder_inp = decoder_inp + self.forward_energy(energy_inp, energy, ret, infer)

        # decoder input
        decoder_inp = decoder_inp
        
        ret["mel_out"] = self.forward_decoder(
            decoder_inp, tgt_nonpadding, emo_cond=stu_emb_2, spk_cond=spk_embed, ret=ret, infer=infer, **kwargs
        )
        
        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                
        return ret

    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, infer=False):
        pitch_padding = mel2ph == 0
        ret["cwt"] = cwt_out = self.pitch_predictor(decoder_inp)
        stats_out = self.cwt_stats_layers(decoder_inp.mean(1))  # [B, 2]
        mean = ret["f0_mean"] = stats_out[:, 0]
        std = ret["f0_std"] = stats_out[:, 1]
        cwt_spec = cwt_out[:, :, :10]
        if infer:
            std = std * self.hparams["cwt_std_scale"]
            f0 = self.cwt2f0_norm(cwt_spec, mean, std, mel2ph)
            assert cwt_out.shape[-1] == 11
            uv = cwt_out[:, :, -1] > 0
        ret["f0_denorm"] = f0_denorm = denorm_f0(f0, uv, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def forward_energy(self, decoder_inp, energy, ret, infer=False):
        ret["energy_pred"] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        energy_embed_inp = energy_pred if infer else energy
        energy_embed_inp = torch.clamp(
            energy_embed_inp * 256 // 4, min=0, max=255
        ).long()
        energy_embed = self.energy_embed(energy_embed_inp)
        return energy_embed

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        _, cwt_scales = get_lf0_cwt(np.ones(10))
        f0 = cwt2f0(cwt_spec, mean, std, cwt_scales)
        f0 = torch.cat([f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), 1)
        f0_norm = norm_f0(f0, None)
        return f0_norm
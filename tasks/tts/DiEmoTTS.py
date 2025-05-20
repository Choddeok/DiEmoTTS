import numpy as np
import torch
import torch.nn.functional as F
from tasks.tts.fastspeech import FastSpeechTask
from utils.plot.plot import spec_to_figure
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import denorm_f0
import torch.nn as nn
import math
from utils.commons.ckpt_utils import load_ckpt, dino_load_ckpt
from utils.nn.model_utils import print_arch, num_params

class ExpressiveFS2Task(FastSpeechTask):

    def forward(self, sample, infer=False, *args, **kwargs):
        hparams = self.hparams
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        
        spk_id = sample.get("spk_ids")
        if hparams['spk_name'] == "0013":
            new_tensor = torch.tensor([2])
            spk_id[0] = new_tensor
        elif hparams['spk_name'] == "0019":
            new_tensor = torch.tensor([8])
            spk_id[0] = new_tensor
            
        emo_id = sample.get("emo_ids")
        if not infer:
            target = sample["mels"]  # [B, T, 80]
            
            global_crop_0 = sample["global_crop_0"]  # [B, T_l, 80]
            global_crop_1 = sample["global_crop_1"]  # [B, T_l, 80]
            local_crop_2 = sample["local_crop_2"]  # [B, T_s, 80]
            local_crop_3 = sample["local_crop_3"]  # [B, T_s, 80]
            local_crop_4 = sample["local_crop_4"]  # [B, T_s, 80]
            local_crop_5 = sample["local_crop_5"]  # [B, T_s, 80]
            
            global_crop_0_lengths = sample["global_crop_0_lengths"]  # [B, T_l, 80]
            global_crop_1_lengths = sample["global_crop_1_lengths"]  # [B, T_l, 80]
            local_crop_2_lengths = sample["local_crop_2_lengths"]  # [B, T_s, 80]
            local_crop_3_lengths = sample["local_crop_3_lengths"]  # [B, T_s, 80]
            local_crop_4_lengths = sample["local_crop_4_lengths"]  # [B, T_s, 80]
            local_crop_5_lengths = sample["local_crop_5_lengths"]  # [B, T_s, 80]
            
            mel2ph = sample["mel2ph"]  # [B, T]
            f0 = sample.get("f0")
            uv = sample.get("uv")
            energy = sample.get("energy")
            output = self.model.compute_loss(
                txt_tokens,
                mel2ph=mel2ph,
                target=target,
                global_crop_0=global_crop_0,
                global_crop_1=global_crop_1,
                local_crop_2=local_crop_2,
                local_crop_3=local_crop_3,
                local_crop_4=local_crop_4,
                local_crop_5=local_crop_5,
                global_crop_0_lengths=global_crop_0_lengths,
                global_crop_1_lengths=global_crop_1_lengths,
                local_crop_2_lengths=local_crop_2_lengths,
                local_crop_3_lengths=local_crop_3_lengths,
                local_crop_4_lengths=local_crop_4_lengths,
                local_crop_5_lengths=local_crop_5_lengths,
                spk_id=spk_id,
                emo_id=emo_id,
                f0=f0,
                uv=uv,
                energy=energy,
                it=self.global_step,
                infer=False,
            )
            losses = {}
            self.add_mel_loss(output["mel_out"], target, losses)
            self.add_dur_loss(output["dur"], mel2ph, txt_tokens, losses=losses)
            self.add_pitch_loss(output, sample, losses)
            self.add_energy_loss(output, sample, losses)
            
            
            epoch = (self.global_step * hparams["max_sentences"]) // 12155
            nepoch = math.ceil((160000 * hparams["max_sentences"]) / 12155)
            dino_loss = DINOLoss(
                hparams["num_emo"],
                6,  # total number of crops = 2 global crops + local_crops_number
                hparams["warmup_teacher_temp"],
                hparams["teacher_temp"],
                hparams["warmup_teacher_temp_epochs"],
                nepoch).cuda()
            enc_loss = EncLoss(
                6).cuda()
            losses["dino_loss"] = dino_loss(output["student_head"], output["teacher_head"], epoch)
            losses["cs_loss"] = enc_loss(output["student_emb"], output["teacher_emb"])
            return losses, output
        else:
            target = sample["mels"]  # [B, T, 80]
            output = self.model(
                txt_tokens,
                target=target,
                spk_id=spk_id,
                emo_id=emo_id,
                infer=True,
            )
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], model_out = self(sample)
        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        outputs = tensors_to_scalars(outputs)

        valid_plots_num = self.hparams['valid_plots']
        for num in valid_plots_num:
            if self.global_step % self.hparams['valid_infer_interval'] == 0 \
                    and batch_idx == num:
                self.save_valid_result(sample, batch_idx, model_out)
                
        return outputs

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = self.hparams["audio_sample_rate"]
        f0s = None
        mel_out = model_out["mel_out"]

        if batch_idx < 2:
            emo = "Neutral"
        elif batch_idx < 22:
            emo = "Angry"
        elif batch_idx < 42:
            emo = "Happy"
        elif batch_idx < 62:
            emo = "Sad"
        elif batch_idx < 82:
            emo = "Surprise"
        elif batch_idx < 402:
            emo = "Neutral"
        elif batch_idx < 422:
            emo = "Angry"
        elif batch_idx < 442:
            emo = "Happy"
        elif batch_idx < 462:
            emo = "Sad"
        else:
            emo = "Surprise"
        
        if self.hparams["plot_f0"]:
            f0_gt = denorm_f0(sample["f0"][0].cpu(), sample["uv"][0].cpu())
            f0_pred = model_out["f0_denorm"]
            f0s = {"GT": f0_gt, "Pred": f0_pred}
        self.plot_mel(
            batch_idx,
            [sample["mels"][0], mel_out[0]],
            name=f"mel_{batch_idx}_{emo}",
            title=f"mel_{batch_idx}_{emo}",
            f0s=f0s,
        )

        wav_pred = self.vocoder.spec2wav(mel_out[0].cpu())
        self.logger.add_audio(f"wav_pred_{batch_idx}_{emo}", wav_pred, self.global_step, sr)

        # gt wav
        if self.global_step <= self.hparams["valid_infer_interval"]:
            mel_gt = sample["mels"][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.logger.add_audio(f"wav_gt_{batch_idx}_{emo}", wav_gt, self.global_step, sr)


    def plot_cwt(self, batch_idx, cwt_out, cwt_gt=None):
        if len(cwt_out.shape) == 3:
            cwt_out = cwt_out[0]
        if isinstance(cwt_out, torch.Tensor):
            cwt_out = cwt_out.cpu().numpy()

        if len(cwt_gt.shape) == 3:
            cwt_gt = cwt_gt[0]
        if isinstance(cwt_gt, torch.Tensor):
            cwt_gt = cwt_gt.cpu().numpy()
        cwt_out = np.concatenate([cwt_gt, cwt_out], -1)
        name = f"plot_cwt_{batch_idx}"
        self.logger.add_figure(name, spec_to_figure(cwt_out), self.global_step)

    def add_pitch_loss(self, output, sample, losses):
        cwt_spec = sample[f"cwt_spec"]
        f0_mean = sample["f0_mean"]
        uv = sample["uv"]
        mel2ph = sample["mel2ph"]
        f0_std = sample["f0_std"]
        cwt_pred = output["cwt"][:, :, :10]
        f0_mean_pred = output["f0_mean"]
        f0_std_pred = output["f0_std"]
        nonpadding = (mel2ph != 0).float()
        losses["f0_cwt"] = F.l1_loss(cwt_pred, cwt_spec) * self.hparams["lambda_f0"]

        assert output["cwt"].shape[-1] == 11
        uv_pred = output["cwt"][:, :, -1]
        losses["uv"] = (
            (
                F.binary_cross_entropy_with_logits(uv_pred, uv, reduction="none")
                * nonpadding
            ).sum()
            / nonpadding.sum()
            * self.hparams["lambda_uv"]
        )
        losses["f0_mean"] = F.l1_loss(f0_mean_pred, f0_mean) * self.hparams["lambda_f0"]
        losses["f0_std"] = F.l1_loss(f0_std_pred, f0_std) * self.hparams["lambda_f0"]

    def add_energy_loss(self, output, sample, losses):
        energy_pred, energy = output["energy_pred"], sample["energy"]
        nonpadding = (energy != 0).float()
        loss = (
            F.mse_loss(energy_pred, energy, reduction="none") * nonpadding
        ).sum() / nonpadding.sum()
        loss = loss * self.hparams["lambda_energy"]
        losses["energy"] = loss
        

    def test_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return:
        """
        assert (
            sample["txt_tokens"].shape[0] == 1
        ), "only support batch_size=1 in inference"
        outputs = self(sample, infer=True)
        text = sample["text"][0]
        item_name = sample["item_name"][0]
        tokens = sample["txt_tokens"][0].cpu().numpy()
        mel_gt = sample["mels"][0].cpu().numpy()
        mel_pred = outputs["mel_out"][0].cpu().numpy()
        
        mel2ph_item = sample.get("mel2ph")
        if mel2ph_item is not None:
            mel2ph = mel2ph_item[0].cpu().numpy()
        else:
            mel2ph = None

        mel2ph_pred_item = outputs.get("mel2ph")
        if mel2ph_pred_item is not None:
            mel2ph_pred = mel2ph_pred_item[0].cpu().numpy()
        else:
            mel2ph_pred = None
            
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        
        hparams = self.hparams
        spk_name = hparams['spk_name']
        
        if hparams['spk_name'] == "0013":
            spk_name = hparams['spk_name']
            base_fn = spk_name + "_" + item_name
        elif hparams['spk_name'] == "0019":
            spk_name = hparams['spk_name']
            base_fn = spk_name + "_" + item_name
        else:
            base_fn = item_name
        
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        
        audio_sample_rate = self.hparams["audio_sample_rate"]
        out_wav_norm = (self.hparams["out_wav_norm"],)
        mel_vmin = self.hparams["mel_vmin"]
        mel_vmax = self.hparams["mel_vmax"]
        save_mel_npy = self.hparams["save_mel_npy"]

        self.saving_result_pool.add_job(
            self.save_result,
            args=[
                wav_pred,
                mel_pred,
                base_fn,
                gen_dir,
                str_phs,
                mel2ph_pred,
                None,
                audio_sample_rate,
                out_wav_norm,
                mel_vmin,
                mel_vmax,
                save_mel_npy,
            ],
        )
        if self.hparams["save_gt"]:
            gt_name = base_fn + "_gt"
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[
                    wav_gt,
                    mel_gt,
                    gt_name,
                    gen_dir,
                    str_phs,
                    mel2ph,
                    None,
                    audio_sample_rate,
                    out_wav_norm,
                    mel_vmin,
                    mel_vmax,
                    save_mel_npy,
                ],
            )
        # print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            "item_name": item_name,
            "text": text,
            "ph_tokens": self.token_encoder.decode(tokens.tolist()),
            "wav_fn_pred": base_fn,
            "wav_fn_gt": base_fn + "_gt",
        }

class DINOLoss(nn.Module):
    def __init__(self, emo_num, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, emo_num))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        

class EncLoss(nn.Module):
    def __init__(self, ncrops):
        super(EncLoss, self).__init__()
        self.ncrops = ncrops
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, student_out, teacher_out):
        student_out = student_out.chunk(self.ncrops)
        teacher_out = teacher_out.detach().chunk(2)
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = 1 - self.cosine_similarity(student_out[v], q)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
from tasks.tts.speech_base import SpeechBaseTask


class FastSpeechTask(SpeechBaseTask):

    def forward(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")
        if not infer:
            target = sample["mels"]  # [B, T_s, 80]
            mel2ph = sample["mel2ph"]  # [B, T_s]
            output = self.model(
                txt_tokens,
                mel2ph=mel2ph,
                spk_embed=spk_embed,
                spk_id=spk_id,
                infer=False,
            )
            losses = {}
            self.add_mel_loss(output["mel_out"], target, losses)
            self.add_dur_loss(output["dur"], mel2ph, txt_tokens, losses=losses)
            return losses, output
        else:
            output = self.model(
                txt_tokens,
                spk_embed=spk_embed,
                spk_id=spk_id,
                infer=True,
            )
            return output

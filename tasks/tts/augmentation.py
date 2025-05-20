import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

class Augment(nn.Module):
    """Waveform Augmentation with only Formant Shifting."""
    def __init__(self):
        """Initializer."""
        super().__init__()
        self.coder = LinearPredictiveCoding(32, 1024, 256)
        self.register_buffer('window', torch.hann_window(1024), persistent=False)

    def forward(self, wavs: torch.Tensor, mode: str = 'linear'):
        """Augment the audio signal using only formant shifting."""
        # [B, F, T / S], complex64
        fft = torch.stft(wavs, 1024, 256, 1024, self.window, return_complex=True)

        # Formant shifting only (no pitch shifting or PEQ)
        fs_ratio = 1.4  # For formant shifting, you can adjust the ratio

        # Apply formant shifting
        code = self.coder.from_stft(fft / fft.abs().mean(dim=1)[:, None].clamp_min(1e-7))
        filter_ = self.coder.envelope(code)
        source = fft.transpose(1, 2) / (filter_ + 1e-7)
        
        # Sample formant shift factors
        bsize = wavs.shape[0]
        def sampler(ratio):
            shifts = torch.rand(bsize, device=wavs.device) * (ratio - 1.) + 1.
            flip = torch.rand(bsize) < 0.5
            shifts[flip] = shifts[flip] ** -1
            return shifts
        
        fs_shift = sampler(fs_ratio)

        # Interpolate for formant shift
        filter_ = self.interp(filter_, fs_shift, mode=mode)
        source = self.interp(source, fs_shift, mode=mode)
       
        fft = (source * filter_).transpose(1, 2)
        out = torch.istft(fft, 1024, 256, 1024, self.window)
        out = out / out.max(dim=-1, keepdim=True).values.clamp_min(1e-7)
        return out

    @staticmethod
    def complex_interp(inputs: torch.Tensor, *args, **kwargs):
        mag = F.interpolate(inputs.abs(), *args, **kwargs)
        angle = F.interpolate(inputs.angle(), *args, **kwargs)
        return torch.polar(mag, angle)

    def interp(self, inputs: torch.Tensor, shifts: torch.Tensor, mode: str):
        """Interpolate the channel axis with dynamic shifts."""
        INTERPOLATION = {
            torch.float32: F.interpolate,
            torch.complex64: Augment.complex_interp}
        assert inputs.dtype in INTERPOLATION, 'unsupported interpolation'
        interp_fn = INTERPOLATION[inputs.dtype]
        _, _, channels = inputs.shape
        
        interp = [interp_fn(f[None], scale_factor=s.item(), mode=mode)[..., :channels]
                    for f, s in zip(inputs, shifts)]
       
        return torch.cat([F.pad(f, [0, channels - f.shape[-1]]) for f in interp], dim=0)

class LinearPredictiveCoding(nn.Module):
    """ LPC: Linear-predictive coding supports. """
    def __init__(self, num_code: int, windows: int, strides: int):
        """Initializer."""
        super().__init__()
        self.num_code = num_code
        self.windows = windows
        self.strides = strides

    def forward(self, inputs: torch.Tensor):
        """Compute the linear-predictive coefficients from inputs."""
        w = self.windows
        frames = F.pad(inputs, [0, w]).unfold(-1, w, self.strides)
        corrcoef = LinearPredictiveCoding.autocorr(frames)
      
        return LinearPredictiveCoding.solve_toeplitz(
            corrcoef[..., :self.num_code + 1])

    def from_stft(self, inputs: torch.Tensor):
        """Compute the linear-predictive coefficients from STFT."""
        corrcoef = torch.fft.irfft(inputs.abs().square(), dim=1)
        return LinearPredictiveCoding.solve_toeplitz(
            corrcoef[:, :self.num_code + 1].transpose(1, 2))

    def envelope(self, lpc: torch.Tensor):
        """LPC to spectral envelope."""
        denom = torch.fft.rfft(-F.pad(lpc, [1, 0], value=1.), self.windows, dim=-1).abs()
        denom[(denom.abs() - 1e-7) < 0] = 1.
        return denom ** -1

    @staticmethod
    def autocorr(wavs: torch.Tensor):
        """Compute the autocorrelation."""
        fft = torch.fft.rfft(wavs, dim=-1)
        return torch.fft.irfft(fft.abs().square(), dim=-1)

    @staticmethod
    def solve_toeplitz(corrcoef: torch.Tensor):
        """Solve the toeplitz matrix."""
        solutions = F.pad(
            (-corrcoef[..., 1] / corrcoef[..., 0].clamp_min(1e-7))[..., None],
            [1, 0], value=1.)
        
        extra = corrcoef[..., 0] + corrcoef[..., 1] * solutions[..., 1]
        num_code = corrcoef.shape[-1] - 1
        for k in range(1, num_code):
            lambda_value = (-solutions[..., :k + 1]
                                * torch.flip(corrcoef[..., 1:k + 2], dims=[-1])
                            ).sum(dim=-1) / extra.clamp_min(1e-7)
            aug = F.pad(solutions, [0, 1])
            solutions = aug + lambda_value[..., None] * torch.flip(aug, dims=[-1])
            extra = (1. - lambda_value ** 2) * extra
        return solutions[..., 1:]

import torch.optim as optim
import torch
import torch.nn as nn
import utils
import torch.fft as tfft

def FocalStackShift(init_phase, target_amp, sim_prop, filter, num_iters, 
                    roi_res=(1080, 1920), loss=nn.MSELoss(), lr=0.06):

    # SLM phase
    slm_phase = init_phase.requires_grad_(True)

    # Adam optimizer
    optimizer = optim.Adam([{'params': slm_phase}], lr=lr)

    # Crop target ROI
    target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)
    target_amp = target_amp.permute(1, 0, 2, 3)
    # Define shifts for focal stack
    shifts = [
        (0, 0), (0, -1), (0, -1), (-2, 0), (0, 2), (0, -1), (1, 0), (0, -1), (0, -1)
    ]

    best_loss = 1e10

    for k in range(num_iters):
        optimizer.zero_grad()

        recon_amps = []

        for shift in shifts:
            # Apply shift to the SLM phase
            shifted_phase = torch.roll(slm_phase, shifts=shift, dims=(-2, -1))

            # Convert to complex field
            real, imag = utils.polar_to_rect(torch.ones_like(shifted_phase), shifted_phase)
            slm_field = torch.complex(real, imag)

            # Apply filter and simulate propagation
            slm_field = tfft.fftshift(tfft.fftn(slm_field, dim=(-2, -1), norm='ortho'), (-2, -1))
            slm_field = slm_field * filter
            slm_field = tfft.ifftn(tfft.ifftshift(slm_field, (-2, -1)), dim=(-2, -1), norm='ortho')

            # Compute reconstructed amplitude
            recon_amp = torch.abs(sim_prop(slm_field))
            recon_amps.append(recon_amp)

        # Average reconstructed amplitudes
        recon_amp = torch.mean(torch.stack(recon_amps, dim=0), dim=0)

        # Crop to ROI
        recon_amp = utils.crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)

        # Scale the reconstructed amplitude
        with torch.no_grad():
            scale_factor = (recon_amp * target_amp).mean() / (recon_amp ** 2).mean()

        # Compute loss
        loss_value = loss(scale_factor * recon_amp, target_amp)

        # Backpropagation and optimization
        loss_value.backward()
        optimizer.step()

        with torch.no_grad():
            if loss_value.item() < best_loss:
                best_phase = slm_phase
                best_loss = loss_value.item()
                best_amp = scale_factor * recon_amp 

    return best_phase, best_amp
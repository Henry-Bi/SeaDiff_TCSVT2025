import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusion(nn.Module):
    def __init__(self, model, T, schedule):
        super().__init__()
        self.visual = False
        if self.visual:
            self.num = 0
        self.model = model
        self.T = T
        self.schedule = schedule
        betas = self.schedule.get_betas()
        self.register_buffer("betas", betas.float())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        gammas = alphas_bar

        self.register_buffer("coeff1", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "coeff2", self.coeff1 * (1.0 - alphas) / torch.sqrt(1.0 - alphas_bar)
        )
        self.register_buffer(
            "posterior_var", self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )

        self.register_buffer("gammas", gammas)
        self.register_buffer("sqrt_one_minus_gammas", torch.sqrt(1 - gammas))
        self.register_buffer("sqrt_gammas", torch.sqrt(gammas))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t
            - extract(self.coeff2, t, x_t.shape) * eps
        )

    def predict_eps_from_x0(self, x_t, t, x_0):
        return (x_t - extract(self.sqrt_gammas, t, x_t.shape) * x_0) / extract(
            self.sqrt_one_minus_gammas, t, x_t.shape
        )

    def x0_p_mean_variance(self, x_t, cond_, t):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        x0_pred = self.model(torch.cat((x_t, cond_), dim=1), t)

        eps = self.predict_eps_from_x0(x_t, t, x0_pred)

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def p_mean_variance(self, x_t, cond_, t):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(torch.cat((x_t, cond_), dim=1), t)

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def noisy_image(self, t, y):
        """Compute y_noisy according to (6) p15 of [2]"""
        noise = torch.randn_like(y)
        y_noisy = (
            extract_(self.sqrt_gammas, t, y.shape) * y
            + extract_(self.sqrt_one_minus_gammas, t, noise.shape) * noise
        )
        return y_noisy, noise

    def forward(self, x_T, cond, cond_J, cond_hist, pre_ori="False"):
        """
        Algorithm 2.
        """
        x_t = x_T
        cond_ = cond
        cond_hist_ = cond_hist
        cond_J_ = cond_J
        for time_step in reversed(range(self.T)):
            print("time_step: ", time_step)
            t = (
                x_t.new_ones(
                    [
                        x_T.shape[0],
                    ],
                    dtype=torch.long,
                )
                * time_step
            )
            if pre_ori == "False":
                mean, var = self.p_mean_variance(
                    x_t=x_t, t=t, cond_=torch.cat((cond_, cond_J_, cond_hist_), dim=1)
                )
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.sqrt(var) * noise
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            else:
                mean, var = self.x0_p_mean_variance(
                    x_t=x_t, t=t, cond_=torch.cat((cond_, cond_J_, cond_hist_), dim=1)
                )
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.sqrt(var) * noise
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return x_0


if __name__ == "__main__":
    from schedule import Schedule

    test = GaussianDiffusion(None, 100, Schedule("linear", 100))
    print(test.gammas)

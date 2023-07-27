"""
Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

Written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.
Thanks to Alexander Safonov for advise on PyTorch implementations. 

This algorithm implements quasi-Newton variational Bayes (QNVB) according to the
paper, "Projective Integral Updates for High-Dimensional Variational Inference."
"""

import torch
from torch import Tensor

__all__ = ["Qnvb", "qnvb"]

class Qnvb(torch.optim.Optimizer):
    """ Quasi-Newton Variational Bayes.
    This is an implementation of projective integral updates for Gaussian mean-field variational inference.
    See paper, Projective Integral Updates for High-Dimensional Variational Inference.

    Args:
        params : Model parameters.
        device : torch.device('cpu') or torch.device('cuda')
        quadrature : string for the quadrature method to use {'mc', 'qmc1', 'qmc2', 'hadamard_cross'}.
        num_eval : Number of function evaluations to use. Must be at least 2. Hadamard cross requires 4.
        sigma_min : Lower bound on standard deviations.
        sigma_max : Upper bound on standard deviations.
        scale_min : Relative lower bound on change in standard deviations from current values.
        scale_max : Relative upper bound on change in standard deviations from current values.
        lr : The upper bound on expected parameter perturbations per step. This is active if the quasi-Newton step is too large.
        betas : Running average coefficients used by Adam. Beta1 affects running gradient. Beta2 affects running second moments of gradient and Hessian.
        eps : Adam epsilon coefficient for numerical stability.
        likelihood_weight : This is the effective number of cases in the annealing likelihood function. For an annealing coefficient,
            alpha in the interval 0 to 1, the likelihood weight would be alpha times the number of training cases.

        Usage:
        The following is an example training loop used for ResNet18
            for i, (images, labels) in enumerate(train_loader):
                # Send both the inputs and the labels to the device, i.e. CPU or GPU.
                images = images.to(device)
                labels = labels.to(device)
        
                # Create a wrapper function to evaluate the model without any arguments. This should return a tensor of predictions.
                def model_func():
                    return model(images)
        
                # Create a second wrapper to evaluate the loss function from the outputs above.
                # The criterion function should use an mean reduction over cases in the batch.
                # Given the return values: outputs = model_func(), we have:
                def loss_func(outputs):
                    return criterion(outputs, labels)
        
                # The following command will evaluate the model and automatically backpropagate several times to update the variational distribution.
                loss, outputs = optimizer.step((model_func, loss_func))
        
                # Then this is standard code to track the average loss and accuracy.
                _, max_pred = torch.max(outputs, 1)
                train_loss.add_(loss*labels.size(0))
                train_acc.add_((max_pred == labels).sum())
                train_count.add_(labels.size(0))

         The following example test loop is very similar, but uses the variational predicitive method to compute integrated predicitions:
            for j, (images, labels) in enumerate(test_loader):
                # This is the same as above in the training loop:
                images = images.to(device)
                labels = labels.to(device)
        
                def model_func():
                    return model(images)
        
                def loss_func(outputs):
                    return criterion(outputs, labels)
        
                # This method only evaluates the variational predictive integral for the given inputs.
                outputs = optimizer.evaluate_variational_predictive(model_func)
        
                _, max_pred = torch.max(outputs, 1)
                test_loss.add_(loss_func(outputs)*labels.size(0))
                test_acc.add_((max_pred == labels).sum())
                test_count.add_(labels.size(0))
 
         Annealing can be performed by including the following code at the beginning of each epoch.
         Let likelihood_weight_0 be the initial likelihood weight and likelihood_increase_factor be the factor by which it is multiplied with each new epoch.
             optimizer.set_likelihood_weight(likelihood_weight_0*(likelihood_increase_factor**current_epoch))

    """
    def __init__(
        self,
        params,
        device=torch.device('cuda'),
        num_eval=4,
        quadrature="hadamard_cross",
        sigma_min=1e-5,
        sigma_max=1e-3,
        scale_min=0.99,
        scale_max=1.01,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        likelihood_weight=5e4,
    ):
        if not 2 <= num_eval:
            raise ValueError("Second-order quadratures require at least two evaluations: {}".format(num_eval))
        if not (quadrature == "mc" or quadrature == "qmc1" or quadrature == "qmc2" or quadrature == "hadamard_cross"):
            raise ValueError("Unrecognized quadrature type {}. Use mc, qmc1, qmc2, or hadamard_cross.".format(quadrature))
        if not 0 < sigma_min:
            raise ValueError("The minimum standard deviation {} must be postive.".format(sigma_min))
        if not sigma_min < sigma_max:
            raise ValueError("The maximum standard deviation {} must be greater than the minimum {}.".format(sigma_max, sigma_min))
        if not scale_min < 1:
            raise ValueError("The minimum scaling rate for the standard deviation {} must be less than 1.".format(scale_min))
        if not 1 < scale_max:
            raise ValueError("The maximum scaling rate for the standard deviation {} must be greater than 1.".format(scale_max))
        if not 0 < lr:
            raise ValueError("The initial learning rate must be positive: {}".format(learn_init))
        if not 0 < betas[0] < 1:
            raise ValueError("The average coefficient beta1 must be between 0 and 1: {}".format(beta1))
        if not 0 < betas[1] < 1:
            raise ValueError("The average coefficient beta2 must be between 0 and 1: {}".format(beta2))
        if not 0 < eps:
            raise ValueError("The denominator coefficient eps must be greater than 0: {}".format(eps))
        if not 1 <= likelihood_weight:
            raise ValueError("The likelihood weight, i.e. the number of independent training cases (potentially annealed), must be greater than 1: {}".format(likelihood_weight))
        # Set hyperparamenters that can be customized for each parameter group.
        defaults = dict(sigma_min=torch.tensor(sigma_min, device=device),
                        sigma_max=torch.tensor(sigma_max, device=device),
                        scale_min=torch.tensor(scale_min, device=device),
                        scale_max=torch.tensor(scale_max, device=device),
                        lr=torch.tensor(lr, device=device))
        super().__init__(params, defaults)

        # Compute device
        self.device = device

        # Running average and annealing attributes. Convert betas to effective sample size.
        self.g_batch_count = torch.tensor(0., device=self.device, requires_grad=False)
        self.h_batch_count = torch.tensor(0., device=self.device, requires_grad=False)
        self.g_batch_target = torch.tensor(1./(1. - betas[0]), device=self.device, requires_grad=False)
        self.h_batch_target = torch.tensor(1./(1. - betas[1]), device=self.device, requires_grad=False)
        self.eps = torch.tensor(eps, device=self.device, requires_grad=False)
        self.likelihood_weight = torch.tensor(likelihood_weight, device=self.device, requires_grad=False)

        # Quadrature attributes
        self.quadrature = quadrature
        self.num_eval = torch.tensor(num_eval, device=self.device, requires_grad=False)
        if quadrature == "hadamard_cross" and not torch.log2(self.num_eval) == torch.floor(torch.log2(self.num_eval)):
            raise ValueError("The Hadamard cross-polytope quadratures require an integer power of 2 evaluations: {}".format(num_eval))

        # Initialize group and parameter variables
        self.num_par = torch.tensor(0, device=self.device)
        for grp in self.param_groups:
            for p in grp['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # Add newly initialized trainable parameters to total.
                    self.num_par.add_(p.numel())
                    # Mean
                    state['mu'] = p.detach().clone()
                    state['mu'].requires_grad = False
                    # Standard Deviation
                    state['sigma'] = torch.ones_like(p, memory_format=torch.preserve_format, requires_grad=False).mul_(grp['sigma_min'])
                    # Running training gradient average
                    state['gt'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
                    # Running training hessian average
                    state['ht'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
                    # Running second moment average
                    state['st'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
        print("num_par = {}".format(self.num_par))
        self.quad_iter = torch.tensor(0, dtype=torch.int64, device=self.device, requires_grad=False)
 
    def initial_state_str(self):
        ret_str = "QNVB 2.4:"
        ret_str += " num_eval={}".format(self.num_eval)
        ret_str += " quadrature={}".format(self.quadrature)
        ret_str += " g_batch_count={:.0f}/{:.0f}".format(self.g_batch_count, self.g_batch_target)
        ret_str += " h_batch_count={:.0f}/{:.0f}".format(self.h_batch_count, self.h_batch_target)
        ret_str += " eps={}".format(self.eps)
        ret_str += " likelihood_weight={}".format(self.likelihood_weight)
        for i, grp in enumerate(self.param_groups):
            ret_str += " group_{}: [".format(i)
            ret_str += " sigma_min={:.1e}".format(grp['sigma_min'])
            ret_str += " sigma_max={:.1e}".format(grp['sigma_max'])
            ret_str += " scale_min={:.3f}".format(grp['scale_min'])
            ret_str += " scale_max={:.3f}".format(grp['scale_max'])
            ret_str += " lr={:.3f}".format(grp['lr'])
            ret_str += " ]"
        return ret_str

    def set_likelihood_weight(self, likelihood_weight):
        self.likelihood_weight = torch.tensor(likelihood_weight, device=self.device, requires_grad=False)

    def step(self, closure):
        """ Step function for training
            Args:
                closure = (model_func, loss_func)
                    model_func() : This function evaluates the model and returns the outputs.
                    loss_func(outputs) : This function accepts model outputs and evaluates the loss criterion.

            Notes:
                Both functions are needed to evaluate the gradient at quadrature points.
                Neither model_func() nor loss_func(outputs) need to zero gradients or call backpropagation.
                These operations are automatically handled as needed. """
        self._cuda_graph_capture_health_check()
        (model_func, loss_func) = closure
        loss_vp, outputs = self._loss_quad(model_func, loss_func)
        self._add_train()
        self._update_density()
        return loss_vp, outputs

    def evaluate_variational_predictive(self, closure):
        # closure = model_func : This function evaluates the model and returns the outputs.
        self._cuda_graph_capture_health_check()

        self._init_quad()
        outputs_ = []
        rng_state = torch.random.get_rng_state()
        for quad_index in range(self.num_eval):
            # Set parameters to quadrature evaluation point.
            self._set_quad(quad_index)

            # Evaluate the model.
            torch.random.set_rng_state(rng_state)
            outputs = closure()

            if len(outputs_) == 0:
                outputs_ = outputs.detach()
            else:
                outputs_.add_(outputs.detach())

        # Average to obtain the variational-predictive outputs.
        outputs_.div_(self.num_eval)
        return outputs_
 

# ======================
# Quadrature subroutines
# ======================
    def simplex(self):
        b = self.num_eval - 1
        s = torch.sqrt(b + 1.)
        a = 1./(1. + s)
        X_ = torch.cat((torch.eye(b, device=self.device).mul_(s).sub_(a), -torch.ones((1, b), device=self.device)), 0)
        X_.requires_grad = False
        return X_

    def hadamard_cross(self):
        jc = 0
        num_ind = self.num_eval.div(2).int()
        quad_ind = torch.arange(self.quad_iter, self.quad_iter + num_ind, dtype=torch.int32, device=self.device, requires_grad=False).reshape((num_ind, 1))
        one = torch.tensor(1, dtype=torch.int32, device=self.device, requires_grad=False)
        self.quad_iter.add_(num_ind)
        for grp in self.param_groups:
            for p in grp['params']:
                if p.requires_grad:
                    state = self.state[p]
                    np = p.numel()
                    B = torch.bitwise_and(torch.arange(jc, jc + np, dtype=torch.int32, device=self.device).reshape((1, np)), quad_ind)
                    B.requires_grad = False
                    num_bit = torch.tensor(32, dtype=torch.int32, device=self.device, requires_grad=False)
                    while num_bit > 1:
                        num_bit >>= 1
                        B.bitwise_xor_(B.bitwise_right_shift(num_bit))
                    B.bitwise_and_(one).bitwise_left_shift_(1).sub_(1)
                    state['X'] = B.float()
                    state['X'].requires_grad = False
                    jc = jc + np

    def monte_carlo(self, style):
        for grp in self.param_groups:
            for p in grp['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['X'] = torch.randn((self.num_eval, p.numel()), dtype=p.dtype, device=p.device, requires_grad=False)
                    if style == "qmc1" or style == "qmc2":
                        # Set mean to zero.
                        state['g_'] = state['X'].mean(0)
                        state['X'].sub_(state['g_'])
                    if style == "qmc2":
                        # Set second moment to one.
                        state['g_'] = (state['X']**2).mean(0)
                        state['X'].div_(state['g_'].sqrt())

    def _init_quad(self):
        if self.quadrature == "hadamard_cross":
            self.hadamard_cross()
        else:
            self.monte_carlo(self.quadrature)
        for grp in self.param_groups:
            for p in grp['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # Initialize quadrature gradient and Hessian diagonal accumulators.
                    state['g_'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
                    state['h_'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)

    def _set_quad(self, quad_index):
        if self.quadrature == "hadamard_cross":
            x_index = quad_index//2
            value = (quad_index % 2)*2 - 1
        else:
            x_index = quad_index
            value = 1
        for grp in self.param_groups:
            for p in grp['params']:
                if p.requires_grad:
                    state = self.state[p]
                    p.data = torch.addcmul(state['mu'], state['sigma'], state['X'][x_index, :].reshape(p.size()), value=value)

    def _acc_quad(self, quad_index):
        if self.quadrature == "hadamard_cross":
            x_index = quad_index//2
            value = (quad_index % 2)*2 - 1
        else:
            x_index = quad_index
            value = 1
        for grp in self.param_groups:
            for p in grp['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['g_'].add_(p.grad)
                    state['h_'].add_(p.grad.mul(state['X'][x_index, :].reshape(p.size())), alpha=value)

    def _loss_quad(self, model_func, loss_func):
        outputs_ = []
        self.j_ = torch.tensor(0., device=self.device)
        self._init_quad()
        model_rng_state = torch.random.get_rng_state()
        for quad_index in range(self.num_eval):
            # Set parameters to quadrature evaluation point.
            self._set_quad(quad_index)

            # Evaluate the model.
            self.zero_grad()
            torch.random.set_rng_state(model_rng_state)
            outputs = model_func()

            # Evaluate the loss function and gradient.
            loss = loss_func(outputs)
            loss.backward()
            self.j_.add_(loss)

            # Accumulate quadrature quantities.
            self._acc_quad(quad_index)
            if len(outputs_) == 0:
                outputs_ = outputs.detach()
            else:
                outputs_.add_(outputs.detach())

        # Evaluate the variational-predictive loss.
        self.j_.div_(self.num_eval)
        outputs_.div_(self.num_eval)
        return loss_func(outputs_), outputs_


# ==============================
# Variational update subroutines
# ==============================
    def _add_train(self):
        with torch.no_grad():
            self.g_batch_count = torch.minimum(self.g_batch_target, self.g_batch_count + 1)
            self.h_batch_count = torch.minimum(self.h_batch_target, self.h_batch_count + 1)
            g_beta = torch.maximum(torch.tensor(0., device=self.device), (self.g_batch_count - 1)/self.g_batch_count) 
            h_beta = torch.maximum(torch.tensor(0., device=self.device), (self.h_batch_count - 1)/self.h_batch_count)
            g_alpha = (1. - g_beta)/self.num_eval 
            h_alpha = (1. - h_beta)/self.num_eval 
            s_alpha = (1. - h_beta)/self.num_eval**2 
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        state['gt'].mul_(g_beta).add_(state['g_'], alpha=g_alpha)
                        state['st'].mul_(h_beta).addcmul_(state['g_'], state['g_'], value=s_alpha)
                        state['h_'].div_(state['sigma'])
                        state['ht'].mul_(h_beta).addcmul_(state['h_'], state['h_'], value=s_alpha)

    def _update_density(self):
        with torch.no_grad():
            for grp in self.param_groups:
                for (i, p) in enumerate(grp['params']):
                    if p.requires_grad:
                        state = self.state[p]
                        # Using h_ as the effective hessian.
                        state['h_'] = state['ht'].sqrt()
                        # Apply scaling rate bounds and absolute bounds.
                        state['sigma'] = torch.maximum(state['sigma'].mul(grp['scale_min']),
                                         torch.minimum(state['sigma'].mul(grp['scale_max']), state['h_'].mul(self.likelihood_weight).rsqrt()))
                        state['sigma'] = torch.maximum(grp['sigma_min'], torch.minimum(state['sigma'], grp['sigma_max']))
                        # Using g_ as a temporary variable for change in mu. Maximum expected change is lr*sigma_max.
                        state['g_'] = state['gt'].div(torch.maximum(state['h_'], (state['st'].sqrt() + self.eps)/grp['lr']))
                        # Update mean and average gradient.
                        state['mu'].sub_(state['g_'])
                        state['gt'].addcmul_(state['h_'], state['g_'], value=-1)

    def export_mu(self):
        with torch.no_grad():
            mu = torch.zeros((self.num_par, 1), device=self.device)
            jc = 0
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        np = p.numel()
                        mu[jc:jc+np] = state['mu'].reshape((np, 1))
                        jc = jc + np
        return mu

    def export_sigma(self):
        with torch.no_grad():
            sigma = torch.zeros((self.num_par, 1), device=self.device)
            jc = 0
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        np = p.numel()
                        sigma[jc:jc+np] = state['sigma'].reshape((np, 1))
                        jc = jc + np
        return sigma

    def import_mu(self, mu):
        with torch.no_grad():
            jc = 0
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        np = p.numel()
                        state['mu'] = mu[jc:jc+np].reshape(p.size())
                        jc = jc + np

    def import_sigma(self, sigma):
        with torch.no_grad():
            jc = 0
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        np = p.numel()
                        state['sigma'] = sigma[jc:jc+np].reshape(p.size())
                        jc = jc + np



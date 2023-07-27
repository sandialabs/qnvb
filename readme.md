# Readme for QNVB
QNVB is an implemenatation of projective integral updates for Gaussian mean-field variational inference that is intended to calibrate and control model uncertainty during optimization.

This PyTorch implementation inherits from the Optim class of training optimizers and was programmed for Python 2.0 and tested on single-node with a GPU.
See the paper, 'Projective Integral Updates for High-Dimensional Variational Inference,' for additional information about the sign of this algoirthm.

## Environment Settings
This project was written and tested using:
    Python 3.8.6, [GCC 7.5.0] on linux
    NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7
    Torch 2.0.0
    TorchVision 0.15.1

## Training Usage
One significant difference between the step() proceedure for QNVB versus other optimizers is the need to include a closure comprising both model evaluations and loss evaluations.
This is because QNVB needs to be able to evaluate the model and loss gradient several times to compute the relevant quadrature formulas. An example of the main training loop for ResNet18 follows:

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

        # Then this is standard code to track the total loss, accuracy count, and the number of cases seen.
        _, max_pred = torch.max(outputs, 1)
        train_loss.add_(loss*labels.size(0))
        train_acc.add_((max_pred == labels).sum())
        train_count.add_(labels.size(0))

## Testing Usage
Validation or testing code is very similar, but uses the variational predicitive method to compute integrated predicitions over the variational density:

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

## Annealing
Also note that variational annealing can be performed by setting the likelihood weight, which is the annealing coefficient alpha mutliplied by the effective number of cases n_t.
That is, likelihood_weight = alpha*n_t. To update the likelihood weight during training, for example at the begining of each epoch, use:

    # Let likelihood_weight_0 be the initial likelihood weight and likelihood_increase_factor be the factor by which it is multiplied with each new epoch.
    # Then annealing is performed by including the following like at the beginning of each epoch.
    optimizer.set_likelihood_weight(likelihood_weight_0*(likelihood_increase_factor**current_epoch))



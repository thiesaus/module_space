

gradients = []
activations = []


def backward_hook(module, grad_input, grad_output):
    print(f"Backward hook running...{module}")

    global gradients # refers to the variable in the global scope
    #print('Backward hook running...')
    gradients = grad_output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    #print(f'Gradients size: {gradients[0].size()}')
    # We need the 0 index because the tensor containing the gradients comes
    # inside a one element tuple.

def forward_hook(model, args, output):
    print(f"Forward hook running...{output}")
    global activations # refers to the variable in the global scope
    #print(f'Module: {module}')
    #print('Forward hook running...')
    activations = output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    #print(f'Activations size: {activations.size()}')

def test(model):
    # self.model.register_forward_hook(self.forward_hook, prepend=False)
    for module in model.modules():
        module.register_full_backward_hook(backward_hook, prepend=False)
        print('added hook to', module)
        
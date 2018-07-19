import torch

def train(model,x,y,criterion,opt,epochs,silent=True):
    losses = []
    model.train()
    for epoch in range(epochs):
        if not silent:
            if epoch % 100 == 100 - 1:
                print(epoch+1) # progress update
        #for batch in train_generator:
        model.zero_grad()
        outputs = torch.stack(model(x))
        #last_outputs = torch.stack(outputs[:-1])
        loss = criterion(outputs.squeeze(), y.squeeze())

        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

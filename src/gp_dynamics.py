import torch
import time
import os
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
# from livelossplot import PlotLosses
import matplotlib.pyplot as plt
from IPython import display


class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_inputs, num_tasks, num_inducing_pts, inducing_pts=None, mean='zero'):
        self.num_inputs = num_inputs
        self.num_tasks = num_tasks
        self.num_inducing_pts = num_inducing_pts
        
        if inducing_pts is None:
            inducing_points = torch.rand(num_tasks, num_inducing_pts, num_inputs)
        else:
            inducing_points = inducing_pts
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )
        
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )
        
        super().__init__(variational_strategy)
        
        if mean == 'zero':
            self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_tasks]))
        elif mean == 'constant':
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        elif mean == 'linear':
            self.mean_module = gpytorch.means.LinearMean(num_inputs, batch_shape=torch.Size([num_tasks]))
        else:
            raise ValueError('Invalid mean. Choises are: "zero", "constant", and "linear".')
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=num_inputs, batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class VariationalGPDynamics():
    def __init__(self, nx, nu, num_inducing_pts=16, inducing_pts=None, mean='zero', model_path='vgp_model.pt', device='cpu'):
        self.nx = nx
        self.nu = nu
        self.device = device
        self.model = IndependentMultitaskGPModel(nx+nu, nx, num_inducing_pts, inducing_pts=inducing_pts, mean=mean)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nx)
        self.model_path = model_path
        
        if not device.type == 'cpu':
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        
        # Save some training statistics
        self.loss_hist = []
        self.start_epoch = 0
    
    def train(self, train_loader, num_epochs=10, lr=0.01, verbose=True, plot_loss=False, num_batches=None):
        self.model.train()
        self.likelihood.train()
        
        # Use Adam optimizer for stochastic gradient descent
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=lr)
        
#         mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(train_loader.dataset))
        mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.model, num_data=len(train_loader.dataset))

        if plot_loss:
            fig = plt.figure(figsize=(8,4), facecolor='white')
            fig.suptitle('Negative Log Likelihood', fontsize=12)
            ax = fig.add_subplot(111)
            if len(self.loss_hist) == 0:
                ax.plot(self.loss_hist)
#                 display.display(fig)
        else:
                ax.plot(self.loss_hist)
#                 display.display(fig)
        
        if num_batches is None:
            num_batches = len(train_loader)
        assert num_batches <= len(train_loader)
        
        for epoch in range(self.start_epoch, num_epochs):
            logs = {}
            
            total_loss = 0
            start_time = time.time()
            for i, data in enumerate(train_loader):
                if i+1 > num_batches:
                    break
                x = data['x'].to(self.device)
                u = data['u'].to(self.device)
                dx = data['dx'].to(self.device)
                
                optimizer.zero_grad()
                output = self.model(torch.cat((x, u), dim=-1))
                loss = -mll(output, dx)
                loss.backward()
                optimizer.step()
        
                total_loss += loss
            
            total_loss /= len(train_loader)
            
            self.loss_hist.append(total_loss.item())
            self.start_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch + 1,
                'loss_hist': self.loss_hist,
                'model_state_dict': self.model.state_dict(),
                'likelihood_state_dict': self.likelihood.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, self.model_path)
            
            if plot_loss:
                display.clear_output(wait=True)
                ax.clear()
                ax.plot(self.loss_hist)
#                 line.set_xdata(range(len(self.loss_hist)))
#                 line.set_ydata(self.loss_hist)
                display.display(fig)
#                 logs['log_loss'] = total_loss.item()
#                 liveloss.update(logs)
#                 liveloss.send()
#                 ax.plot(self.loss_hist)

#                 plt.show()
        
            print(f"Epoch: {epoch+1}/{num_epochs} | Loss: {total_loss} | Time: {time.time() - start_time}")
        
        print('Finished Training')
    
    def load(self, model_path):
        if os.path.isfile(model_path):
            print(f"=> loading model checkpoint '{model_path}'")
            checkpoint = torch.load(model_path, map_location=torch.device(self.device))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
            self.loss_hist = checkpoint['loss_hist']
            self.start_epoch = checkpoint['epoch']
            print(f"=> loaded checkpoint '{model_path}'")
                  
        else:
            print(f"=> no checkpoint found at '{model_path}'")
    
    def predict(self, x, u):
        x = x.reshape(-1,self.nx).to(self.device)
        u = u.reshape(-1,self.nu).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(torch.cat((x, u), dim=-1)))
            mean = x + predictions.mean
            lower, upper = predictions.confidence_region()
            covar = predictions.covariance_matrix
        return mean.cpu(), covar.cpu(), lower.cpu(), upper.cpu()
    
    def mean_step(self, x, u):
        numpy = False
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
            u = torch.from_numpy(u).float()
            numpy = True
            
        x = x.reshape(-1,self.nx).to(self.device)
        u = u.reshape(-1,self.nu).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_next = x + self.likelihood(self.model(torch.cat((x, u), dim=-1))).mean
            
        if numpy:
            return x_next[0].cpu().numpy()
        else:
            return x_next[0].cpu()
    
    def covariance(self, x, u):
        numpy = False
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
            u = torch.from_numpy(u).float()
            numpy = True
            
        x = x.reshape(-1,self.nx).to(self.device)
        u = u.reshape(-1,self.nu).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            covariance_matrix = self.model(torch.cat((x, u), dim=-1)).covariance_matrix
#           covariance_matrix = self.likelihood(self.model(torch.cat((x, u), dim=-1))).covariance_matrix
            
        if numpy:
            return covariance_matrix.cpu().numpy()
        else:
            return covariance_matrix.cpu()
    
    def dx_with_grad(self, x, u):
        x = x.reshape(-1,self.nx).to(self.device)
        u = u.reshape(-1,self.nu).to(self.device)
        return self.likelihood(self.model(torch.cat((x, u), dim=-1))).mean
        
    def linearize(self, x, u):        
        numpy = False
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
            u = torch.from_numpy(u).float()
            numpy = True
        
        x = x.requires_grad_(True).to(self.device)
        u = u.requires_grad_(True).to(self.device)
        A, B = torch.autograd.functional.jacobian(self.dx_with_grad, inputs=(x,u))
        A += torch.eye(self.nx).to(self.device)
#         mean = self.likelihood(self.model(torch.cat((x, u), dim=-1))).mean
        d = - A.matmul(x) - B.matmul(u) + self.mean_step(x, u)
    
        if numpy:
            return A[0].detach().cpu().numpy(), B[0].detach().cpu().numpy(), d[0].detach().cpu().numpy()
        else:
            return A[0].detach().cpu(), B[0].detach().cpu(), d[0].detach().cpu()
                                        
        
    def eval(self):
        self.model.eval()
        self.likelihood.eval()
                
        




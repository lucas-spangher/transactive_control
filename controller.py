import numpy as np

class BaseController():
    """
    Controller parent class -- defines how to interact with controllers
    """


    def __init__(self):
        pass

    def get_points(self, price_signal, **kwargs):
        """
        Inputs:
            price_signal -

        returns
            points - array of point values
            times - times for which the point values apply
        """
        return price_signal

    def update(self, reward):
        """
        Inputs:
            Some sort of reward that will updates the controller's policy.
        """
        pass


class RLController(BaseController):
    """

    """
    def __init__(self):
        self.policy = SimpleNet()

    def get_points(price_signal):
        return self.policy(price_signal)

    def update(reward):
        pass



# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()

#         self.fc1 = nn.Linear(12, 10)
#         self.fc2 = nn.Linear(10, 10)
#         self.fc3 = nn.Linear(10, 12)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

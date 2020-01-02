"""!
Copyright (C) 2019 Bruno Gregorio - BIPG

    https://brunoggregorio.github.io \n
    https://www.bipgroup.dc.ufscar.br

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from keras.callbacks import Callback
from keras import backend as K
import numpy as np


class CyclicLR(Callback):
    """!@brief
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with some
    constant frequency.

    @param base_lr    : Initial learning rate which is the lower boundary in the cycle.

    @param max_lr     : Upper boundary in the cycle. Functionally, it defines the cycle
                        amplitude (max_lr - base_lr). The lr at any cycle is the sum of
                        base_lr and some scaling of the amplitude; therefore max_lr may
                        not actually be reached depending on scaling function.

    @param step_size  : Number of training iterations per half cycle. Authors suggest
                        setting step_size 2-8 x training iterations in epoch.

    @param mode       : One of {triangular, triangular2, exp_range}. Default
                        'triangular'. Values correspond to policies detailed above.
                        If scale_fn is not None, this argument is ignored.

    @param gamma      : Constant in 'exp_range' scaling function:
                        gamma**(cycle iterations)

    @param scale_fn   : Custom scaling policy defined by a single argument lambda
                        function, where 0 <= scale_fn(x) <= 1 for all x >= 0.
                        Mode paramater is ignored.

    @param scale_mode : {'cycle', 'iterations'}. Defines whether scale_fn is evaluated
                        on cycle number or cycle iterations (training iterations since
                        start of cycle). Default is 'cycle'.

    The amplitude of the cycle can be scaled on a per-iteration or per-cycle basis.
    This class has three built-in policies, as put forth in the paper.

    - "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    - "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    - "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.

    For more detail, please see paper.

    #### Example for CIFAR-10 w/ batch size 100:

    \code{.py}
        clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                            step_size=2000., mode='triangular')
        model.fit(X_train, Y_train, callbacks=[clr])
    \endcode

    Class also supports custom scaling functions:
    \code{.py}
        clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
        clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                            step_size=2000., scale_fn=clr_fn,
                            scale_mode='cycle')
        model.fit(X_train, Y_train, callbacks=[clr])
    \endcode

    #### References
      - [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
      - https://www.pyimagesearch.com/2019/07/29/cyclical-learning-rates-with-keras-and-deep-learning/
      - https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/cyclical_learning_rate.py
    """

    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2000.,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """!@brief
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

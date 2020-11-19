#! /usr/bin/env python

# Copyright (c) 2014 KU Leuven, ESAT-STADIUS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import operator as op
import random
import array
import functools

from .solver_registry import register_solver
from .util import Solver, _copydoc, uniform_in_bounds
from . import util
from .Sobol import Sobol

@register_solver('particle swarm',                                                      # name to register solver with
                 'particle swarm optimization',                                         # one-line description of solver
                 ['Maximizes the function using particle swarm optimization.',          # extensive description and manual of solver
                  ' ',
                  'This is a two-phase approach:',
                  '1. Initialization: randomly initializes num_particles particles.',
                  '   Particles are randomized uniformly within the box constraints.',
                  '2. Iteration: particles move during num_generations iterations.',
                  '   Movement is based on their velocities and mutual attractions.',
                  ' ',
                  'This function requires the following arguments:',
                  '- num_particles: number of particles to use in the swarm',
                  '- num_generations: number of iterations used by the swarm',
                  '- max_speed: maximum speed of the particles in each direction (in (0, 1])',
                  '- box constraints via key words: constraints are lists [lb, ub]', ' ',
                  'This solver performs num_particles*num_generations function evaluations.'
                  ])
class ParticleSwarm(Solver):
    """
    .. include:: /global.rst

    Please refer to |pso| for details on this algorithm.
    """

    class Particle:
        def __init__(self, position, speed, best, fitness, best_fitness):
            """Constructs a Particle."""
            self.position = position
            self.speed = speed
            self.best = best
            self.fitness = fitness
            self.best_fitness = best_fitness

        def clone(self):
            """Clones this Particle."""
            return ParticleSwarm.Particle(position=self.position[:], speed=self.speed[:],
                                          best=self.best[:], fitness=self.fitness,
                                          best_fitness=self.best_fitness)

        def __str__(self):
            string = 'Particle{position=' + str(self.position)
            string += ', speed=' + str(self.speed)
            string += ', best=' + str(self.best)
            string += ', fitness=' + str(self.fitness)
            string += ', best_fitness=' + str(self.best_fitness)
            string += '}'
            return string

    def __init__(self, num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0, **kwargs):
        """
        Initializes a PSO solver.

        :param num_particles: number of particles to use
        :type num_particles: int
        :param num_generations: number of generations to use
        :type num_generations: int
        :param max_speed: maximum velocity of each particle
        :type max_speed: float or None
        :param phi1: parameter used in updating position based on local best
        :type phi1: float
        :param phi2: parameter used in updating position based on global best
        :type phi2: float
        :param kwargs: box constraints for each hyperparameter
        :type kwargs: {'name': [lb, ub], ...}

        The number of function evaluations it will perform is `num_particles`*`num_generations`.
        The search space is rescaled to the unit hypercube before the solving process begins.

        >>> solver = ParticleSwarm(num_particles=10, num_generations=5, x=[-1, 1], y=[0, 2])
        >>> solver.bounds['x']
        [-1, 1]
        >>> solver.bounds['y']
        [0, 2]
        >>> solver.num_particles
        10
        >>> solver.num_generations
        5

        .. warning:: |warning-unconstrained|

        """

        assert all([len(v) == 2 and v[0] <= v[1]        # Check format of bounds given for each hyperparameter.
                    for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
        self._bounds = kwargs                           # len(self.bounds) gives number of hyperparameters considered.
        self._num_particles = num_particles
        self._num_generations = num_generations

        self._sobolseed = random.randint(100,2000)      # random.randint(a,b) returns random integer N such that a <= N <= b.

        # Sobol sequences are an example of quasi-random low-discrepancy sequences. Roughly speaking, the discrepancy
        # of a sequence is low if the proportion of points in the sequence falling into an arbitrary set B is close
        # to proportional to the measure of B, as would happen on average in the case of an equidistributed sequence.

        if max_speed is None:
            max_speed = 0.7 / num_generations
#            max_speed = 0.2 / math.sqrt(num_generations)
        self._max_speed = max_speed
        self._smax = [self.max_speed * (b[1] - b[0])
                        for _, b in self.bounds.items()]# dictionary.items() returns view object displaying (key,value) tuple pair list.
        
        self._smin = list(map(op.neg, self.smax))       # operator.neg(obj) returns obj negated (-obj).

        self._phi1 = phi1
        self._phi2 = phi2

    @property
    def phi1(self):
        return self._phi1

    @property
    def phi2(self):
        return self._phi2

    @property
    def sobolseed(self): return self._sobolseed

    @sobolseed.setter
    def sobolseed(self, value): self._sobolseed = value

    @staticmethod
    def suggest_from_box(num_evals, **kwargs):
        """Create a configuration for a ParticleSwarm solver.

        :param num_evals: number of permitted function evaluations
        :type num_evals: int
        :param kwargs: box constraints
        :type kwargs: {'param': [lb, ub], ...}

        >>> config = ParticleSwarm.suggest_from_box(200, x=[-1, 1], y=[0, 1])
        >>> config['x']
        [-1, 1]
        >>> config['y']
        [0, 1]
        >>> config['num_particles'] > 0
        True
        >>> config['num_generations'] > 0
        True
        >>> solver = ParticleSwarm(**config)
        >>> solver.bounds['x']
        [-1, 1]
        >>> solver.bounds['y']
        [0, 1]

        """
        d = dict(kwargs)
        if num_evals > 1000:
            d['num_particles'] = 100
        elif num_evals >= 200:
            d['num_particles'] = 20
        elif num_evals >= 10:
            d['num_particles'] = 10
        else:
            d['num_particles'] = num_evals
        d['num_generations'] = int(math.ceil(float(num_evals) / d['num_particles']))
        return d

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def num_generations(self):
        return self._num_generations

    @property
    def max_speed(self):
        return self._max_speed

    @property
    def smax(self):
        return self._smax

    @property
    def smin(self):
        return self._smin

    @property
    def bounds(self):
        return self._bounds

    def generate(self):
        """Generate a new Particle."""
        if len(self.bounds) < Sobol.maxdim(): # Optunity supports Sobol sequences in up to 40 dimensions (i.e. 40 hyperparameters).
            sobol_vector, self.sobolseed = Sobol.i4_sobol(len(self.bounds), self.sobolseed)
            vector = util.scale_unit_to_bounds(sobol_vector, self.bounds.values())
        else: vector = uniform_in_bounds(self.bounds)

        part = ParticleSwarm.Particle(position=array.array('d', vector),
                                      speed=array.array('d', map(random.uniform,
                                                                 self.smin, self.smax)),
                                      best=None, fitness=None, best_fitness=None)
        return part

    def updateParticle(self, part, best, phi1, phi2):
        """Propagate particle, i.e. update its speed and position according to current personal and global best."""
        u1 = (random.uniform(0, phi1) for _ in range(len(part.position)))           # Generate phi1 and phi2 random number coeffiecents
        u2 = (random.uniform(0, phi2) for _ in range(len(part.position)))           # for each hyperparameter
        v_u1 = map(op.mul, u1, map(op.sub, part.best, part.position))               # Calculate phi1 and phi2 velocity contributions.      
        v_u2 = map(op.mul, u2, map(op.sub, best.position, part.position))
        part.speed = array.array('d', map(op.add, part.speed,                       # Add up velocity contributions.
                                          map(op.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):                                      # Constrain particle speed to range (smin, smax).
            if speed < self.smin[i]:
                part.speed[i] = self.smin[i]
            elif speed > self.smax[i]:
                part.speed[i] = self.smax[i]
        part.position[:] = array.array('d', map(op.add, part.position, part.speed)) # Add velocity to position to propagate particle.

    def particle2dict(self, particle):                          # Convert particle to dict format {"hyperparameter": particle_position}.
        return dict([(k, v) for k, v in zip(self.bounds.keys(), # self.bound.keys() returns hyperparameter names.
                                            particle.position)])

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):             # f is objective function to be optimized.
        
        # map(function,iterable,...): Return an iterator that applies function to every item
        # of iterable, yielding the results. If additional iterable arguments are passed,
        # function must take that many arguments and is applied to the items from all iterables
        # in parallel. With multiple iterables, the interator stops when the shortest iterable
        # is exhausted.

        @functools.wraps(f)                                     # wrapper function evaluating f
        def evaluate(d):
            return f(**d)

        # Determine whether optimization problem is maximization or minimization problem.
        # The 'optimize' function is a maximizer, so if we want to minimize, we basically
        # maximize -f.

        if maximize:
            fit = 1.0
        else:
            fit = -1.0

        pop = [self.generate() for _ in range(self.num_particles)]          # Randomly generate list of num_particle new particles. 
        # "_" is common usage in python, meaning that the iterator is not needed. Like, running a list of int,
        # using range, what matters is the times the range shows not its value. It is just another variable,
        # but conventionally used to show that one does not care about its value.

        best = None                                                         # Initialize particle storing global best.

        for g in range(self.num_generations):                               # Loop over generations.
            fitnesses = pmap(evaluate, list(map(self.particle2dict, pop)))  # Evaluate fitnesses for all particles in current generation.
            for part, fitness in zip(pop, fitnesses):                       # Loop over pairs of particles and individual fitnesses.
                part.fitness = fit * util.score(fitness)                    # util.score: wrapper around objective function evaluations to get score.
                if not part.best or part.best_fitness < part.fitness:       # Update personal best if required.
                    part.best = part.position
                    part.best_fitness = part.fitness
                if not best or best.fitness < part.fitness:                 # Update global best if required.
                    best = part.clone()
            for part in pop:                                                # Update particle for next generation loop.
                self.updateParticle(part, best, self.phi1, self.phi2)

        return dict([(k, v)                                                 # Return best position for each hyperparameter.
                        for k, v in zip(self.bounds.keys(), best.position)]), None

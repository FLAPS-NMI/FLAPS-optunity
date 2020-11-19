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

import math             # mathematical functions
import numpy            # scientific computing package
import operator as op   # standard operators as functions
import random           # generate pseudo-random numbers
import array            # efficient arrays of numeric values
import functools        # higher-order functions and operations on callable objects
import os               # miscellaneous operating system interfaces
import copy             # shallow and deep copy operations
import pathlib          # object-oriented filesystem paths
import pickle           # Python object serialization
from mpi4py import MPI  # MPI for Python

# optunity imports
from .solver_registry import register_solver
from .util import Solver, _copydoc, uniform_in_bounds, uniform_in_bounds_dyn_PSO, loguniform_in_bounds_dyn_PSO
from . import util
from .Sobol import Sobol
from . import ParticleSwarm

def updateParam(pop_history, num_params=0, func=None, **kwargs):
    """Update/determine obj. func. params according to user-specified function.
    If function is not specified, all params are set to 1.
    :param pop_history: [list] list of dynamic particle lists from all previous generations
    :param num_params:  [int] number of obj. func. params
    :param func:        [function] how to update obj. func. params
    :returns:           [list] list of obj. func. params
    """
    if func is not None:
        fparams = func(pop_history, num_params, **kwargs)
        print("User-specified updateParam() evaluates to", fparams,".")
        return fparams 
    else:
        print("Use default obj. func. params.")
        return numpy.ones(num_params)

def evaluateObjFunc(args, params=None, func=None, **kwargs):
    """Calculate scalar fitness according to obj. func., given its args and params.
    If `func` is None, the  scalar product `args[i]*params[i]` is returned.

    :param args:   [vector] (unweighted) args of/contributions to obj. func.
    :param params: [vector] params of obj. func.
    :param func:   [function] function specifying functional form of obj. func., i.e.
                   how to combine args and params to obtain scalar fitness
    :returns:      [float] obj. func. value (scalar fitness)
    """
    if func is not None and params is not None:
        print("User-specified combineObj() evaluates to", numpy.around(func(args, params, **kwargs), 2), ".")
        return func(args, params, **kwargs)
    else:
        if params is not None:
            if op.ne(len(args), len(params)):
                raise ValueError("If `combine_obj` is not specified, args and params vectors need to have same length.")
            return numpy.dot(params, args)
        else:
            return sum(args)

@register_solver('dynamic particle swarm',                                                          # name to register solver with
                 'dynamic particle swarm optimization',                                             # one-line description
                 ['Optimizes the function using a dynamic variant of particle swarm optimization.', # extensive description and manual
                  'Parameters of the objective function are adapted after each generation',
                  'according to the current state of knowledge. To make use of this func-',
                  'tionality, the user has to specify two additional functions `update_param`',
                  'and `combine_obj` as well as the number of arguments and parameters in the',
                  'objective function.'
                  ' ',
                  'This is a two-phase approach:',
                  '1. Initialization: Randomly initialize num_particles particles uniformly',
                  '                   within the box constraints.',
                  '2. Iteration: Particles move during num_generations iterations based on',
                  '              their velocities and mutual attractions derived from',
                  '              individual and global best fitnesses.',
                  ' ',
                  'This function requires the following arguments:',
                  '- num_particles: number of particles to use in the swarm',
                  '- num_generations: number of generations',
                  '- max_speed: maximum speed of the particles in each direction (in (0, 1])',
                  '- update_param: function specifying how to determine parameters according',
                  '                to current state of knowledge',
                  '- eval_obj: function specifying how to combine unweighted contributions',
                  '            and parameters of objective function to obtain scalar fitness',
                  '- box constraints via key words: constraints are lists [lb, ub]', ' ',
                  'This solver performs num_particles*num_generations function evaluations.'
                  ])

class DynamicPSO(ParticleSwarm):
    """Dynamic particle swarm optimization solver class."""
    class DynamicParticle(ParticleSwarm.Particle):
        """Dynamic particle class."""
        def __init__(self, position, speed, best, fitness, best_fitness, fargs):
            """Construct a dynamic particle.
            :param position: particle position (hyperparameter combination to be tested)
            :param speed: particle speed (direction of movement in hyperparameter space)
            :param best: best particle position so far (considering current and all previous generations)
            :param fitness: particle fitness (according to its original generation)
            :param best_fitness: (personal) best particle fitness so far (considering current and all previous generations)
            :param fargs: vector of unweighted obj. func. terms for this particle
            """
            super().__init__(position, speed, best, fitness, best_fitness)
            self.fargs = fargs
       
        def __str__(self):
            string = 'Particle{position=' + str(self.position)
            string += ', fargs=' +repr(self.fargs)
            string += ', speed=' + str(self.speed)
            string += ', best=' + str(self.best)
            string += ', fitness=' + str(self.fitness)
            string += ', best_fitness=' + str(self.best_fitness)
            string += '}'
            return string

        def clone(self):
            """Clone this dynamic particle."""
            return DynamicPSO.DynamicParticle(position=self.position[:], speed=self.speed[:],
                                              best=self.best[:], fitness=self.fitness,
                                              best_fitness=self.best_fitness, fargs=self.fargs[:])

    def __init__(self, num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0, update_param=None, eval_obj=None, **kwargs):
        """ Initialize a dynamic PSO solver.
        :param num_particles: [int] number of particles in a generation
        :param num_generations: [int] number of generations
        :param max_speed: [float] upper bound for particle velocity
        :param phi1: [float] acceleration coefficient determining impact of each particle's historical best on its movement
        :param phi2: [float] acceleration coefficient determining impact of global best on movement of each particle
        :param update_param: [function] how to determine obj. func. param.s according to current state of knowledge
        :param eval_obj: [function] how to combine obj. func. arg.s and param.s to obtain scalar fitness
        :param **kwargs: box constraints for each hyperparameter as key-worded arguments
        """
        # Check format of bounds given for each hyperparameter.
        assert all([len(v) == 2 and v[0] <= v[1] for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
        self._bounds = kwargs                           # len(self.bounds) gives number of hyperparameters considered.
        self._num_particles = num_particles
        self._num_generations = num_generations

        self._sobolseed = random.randint(100,2000)      # random.randint(a,b) gives random integer N with a <= N <= b.

        if max_speed is None: 
            max_speed = 0.7/num_generations
        self._max_speed = max_speed
        # Calculate min. and max. velocities for each hyperparameter considered.
        self._smax = [self.max_speed * (b[1] - b[0]) for _, b in self.bounds.items()]   # dictionary.items() returns view object displaying (key,value) tuple pair list.
        self._smin = list(map(op.neg, self.smax))                                       # operator.neg(obj) returns obj negated (-obj).

        self._phi1 = phi1
        self._phi2 = phi2

        self._update_param = update_param
        self._eval_obj = eval_obj

    def split_log_uni(self, domains):
        uni = {}
        log = {}
        for key, value in self.bounds.items():
            if domains[key] == "uniform":
                uni[key] = value
            elif domains[key] == "loguniform":
                log[key] = value
        return uni, log

    def generate(self, domains):
        """Generate new dynamic particle."""
        uni, log = self.split_log_uni(domains)
        # uniformly distributed hyperparameters
        if len(uni) < Sobol.maxdim():
            sobol_vector, self.sobolseed = Sobol.i4_sobol(len(uni), self.sobolseed)
            vector_uni = util.scale_unit_to_bounds(sobol_vector, uni.values())
        else:
            vector_uni = uniform_in_bounds(uni)
        # log-uniformly distributed hyperparameters
        vector_log = [] 
        for idx, value in enumerate(log.values()):
            vector_log.append(loguniform_in_bounds_dyn_PSO(value))

        sorted_bounds = {**uni, **log}
        vector_temp = [vector_uni, vector_log]
        flat_vector_temp = [v for l in vector_temp for v in l]
        vector_dict = {}
        for idx, key in enumerate(sorted_bounds.keys()):
            vector_dict[key] = flat_vector_temp[idx]
        vector = []
        for idx, key in enumerate(self.bounds.keys()):
                vector.append(vector_dict[key])

        part = DynamicPSO.DynamicParticle(position=array.array('d', vector),                # random.uniform(a, b) returns a random floating point number N such that
                                      speed=array.array('d', map(random.uniform,            # a <= N <= b for a <= b and vice versa.
                                                                 self.smin, self.smax)),
                                      best=None, fitness=None, best_fitness=None,
                                      fargs=None)
        print("Position", repr(part.position), ", speed", repr(part.speed))
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
        print("Old position:", part.position[:])
        print("Speed:", part.speed[:])
        part.position[:] = array.array('d', map(op.add, part.position, part.speed)) # Add velocity to position to propagate particle.
        print("New position:", part.position[:])
    
    @_copydoc(Solver.optimize)
    def optimize(self, f, domains, num_args_obj, num_params_obj, maximize=False, pmap=map, comm_inter=MPI.COMM_WORLD, comm_intra=MPI.COMM_WORLD, workspace=pathlib.Path.home()):
        """Actual solver implementing dynamic particle swarm optimization.""" 
        # functools.wraps(wrapped): convenience function for invoking update_wrapper() as function decorator when
        # defining a wrapper function. functools.update_wrapper(wrapper, wrapped) updates a wrapper function to look 
        # like the wrapped function.

        @functools.wraps(f) # wrapper function evaluating f
        def evaluate(d):
            """Wrapper function evaluating obj. func. f accepting a dict {"hyperparameter": particle position}."""
            return f(**d)
        
        if maximize:    # Maximization or minimization problem?
            fit = -1.0  # `optimize` function is a minimizer,
        else:           # i.e. to maximize, minimize -f.
            fit = 1.0
       
        # paths for checkpointing and logging
        hist_path      = workspace+"history.p"
        hist_prev_path = workspace+"history_prev.p"
        best_path      = workspace+"best.p"
        best_prev_path = workspace+"best_prev.p"
        log_path       = workspace+"log.log"
        log_backup     = workspace+"#log.log#"
        params_path    = workspace+"params.log"
        params_backup  = workspace+"#params.log#"

        # MPI stuff
        r_inter = comm_inter.Get_rank()
        s_inter = comm_inter.Get_size()
        r_world = MPI.COMM_WORLD.Get_rank()
        s_world = MPI.COMM_WORLD.Get_size()

        
        print(r_inter, "/", s_inter, "(", r_world,"/", s_world,"): Initialize particle for first generation...")
       
        # Restart from checkpoint if exists.
        try: # Most recent checkpoint exists and is functional.
            with open(hist_path,"rb") as histp:
                PART_temp = pickle.load(histp)
            with open(best_path,"rb") as bestp:
                BEST = pickle.load(bestp)
        except Exception as e: # Most recent checkpoint not there or broken.
            print(e)
            try: # Previous checkpoint exists and is functional.
                with open(hist_prev_path,"rb") as histp:
                    PART_temp = pickle.load(histp)
                with open(best_prev_path,"rb") as bestp:
                    BEST = pickle.load(bestp)
            except Exception as e: # Previous checkpoint not there or broken.
                print(e)
                print("CHECKPOINTING: No history found, randomly initiate particle.")
                PART = self.generate(domains) # Randomly initiate particle.
            else:
                print("CHECKPOINTING: Load previous particle history.")
                idx = r_inter - s_inter
                PART = PART_temp[idx]
                self.updateParticle(PART, BEST, self.phi1, self.phi2)
        else:
            print("CHECKPOINTING: Load most recent particle history.")
            idx = r_inter - s_inter
            PART = PART_temp[idx]
            self.updateParticle(PART, BEST, self.phi1, self.phi2)

        part_history = []   # Initialize particle history list for THIS individual particle.
        fparams_history = []# Initialize obj. func. param. history list.
        best = None         # Initialize particle storing global best. MUST BE SHARED AMONG ALL PARTICLES!
        
        if r_inter == 0:
            if os.path.isfile(log_path):
                os.rename(log_path, log_backup)
            if os.path.isfile(params_path):
                os.rename(params_path, params_backup)
            if os.path.isfile(hist_path):
                os.rename(hist_path, hist_prev_path)
        
        print(r_inter,"/",s_inter,"(",r_world,"/",s_world,"): Start dynamic PSO...")
        for g in range(self.num_generations):                                       # Loop over generations.
            print(r_inter,"/", s_inter,"(",r_world,"/",s_world,"): Evaluate blackbox for generation", str(g+1), "...")
            # Evaluate blackbox, i.e. run actual sim. to return obj. func. args for ONE particle.
            # Start workers upon xtc modification to evaluate REF15 score in parallel (top level).

            # Create sim. directory.
            dir_name=""
            for k, v in self.particle2dict(PART).items():
                dir_name += k+"_"+str(v)+"_"
            dir_name=dir_name[0:-1]
            os.chdir(workspace)                     # Change to workspace.
            os.makedirs(dir_name,exist_ok=True)     # Repetition of same particle allowed (exist_ok = True).
            os.chdir(dir_name)                      # Change to sim. directory.
            path = os.getcwd()                      # Get absolute sim. dir. path.
            path = comm_intra.bcast(path, root=0)   # Broadcast path to workers within block.
            print("Folder sucessfully created...")

            # Evaluate blackbox, i.e. run actual sim.
            print("Now blackbox:")
            try:
                PART.fargs = evaluate(self.particle2dict(PART)) # Set obj. func. args as particle attributes.
            except Exception as e:
                print(e)
            else:
                part_history.append(copy.deepcopy(PART))        # Append particle to local history.
                print("Particle",self.particle2dict(PART),", fargs",repr(PART.fargs))
            
            # Gather particle history to update obj. func. params.
            part_history_global = comm_inter.allgather(part_history)                                  # Gather local part_history lists from PSO sim. ranks.
            part_history_global = [ part for part_hist in part_history_global for part in part_hist ] # Flatten part_history_global.
            
            # Update obj. func. params.
            fparams = updateParam(part_history_global, num_params_obj, self._update_param)  # Update obj. func. param.s.
            fparams_history.append(fparams)                                                 # Append current obj. func. param. set to history.
            
            # Recalculate fitnesses of particle for all generations.
            print(r_inter,"/",s_inter,"(",r_world,"/", s_world,"): Re-calculate fitnesses with latest obj. func. params", repr(numpy.around(fparams, 2)), "...")
            PART.fitness = fit * util.score(evaluateObjFunc(PART.fargs[:], fparams[:], self._eval_obj))
            
            for part in part_history:
                part.fitness = fit * util.score(evaluateObjFunc(part.fargs[:], fparams[:], self._eval_obj)) # Calculate fitnesses using most recent obj. func. params.
                print("Particle",self.particle2dict(part),", fitness",part.fitness)
                part.best_fitness = None                                                                    # Reset personal best fitness.
                part.best = None                                                                            # Reset personal best position
                line = " ".join(map("{:>15.4e}".format, part.position))+"  ".join(map("{:>15.4e}".format, part.fargs))+"{:>15.4e}".format(part.fitness)+"\n"
                with open(log_path, "a+") as log: log.writelines(line)
            
            # Initialize best fitness and best positions for particle.
            best_fitness = None
            best_position = None
            
            # Determine pbest.
            print(r_inter,"/", s_inter,"(",r_world,"/",s_world,"): Determine pbest...")
            for part in part_history:
                if best_fitness == None or part.fitness < best_fitness:
                    best_fitness = part.fitness
                    best_position = part.position
            
            # Set pbest for all generations.
            print(r_inter,"/", s_inter,": Set pbest in mono-history...")
            for part in part_history:
                part.best_fitness = best_fitness
                part.best = best_position
            
            # Set pbest for current particle to be propagated.
            print(r_inter,"/", s_inter,": Set pbest for current particle...")
            PART.best_fitness = best_fitness
            PART.best = best_position

            # Allgather part_history arrays containing correct fitnesses and pbests.
            part_history_global = comm_inter.allgather(part_history)
            part_history_global = [ part for part_hist in part_history_global for part in part_hist ] # Flatten part_history_global
           
            # Determine global best.
            print(r_inter,"/", s_inter,"(",r_world,"/",s_world,"): Determine gbest...")
            best = None
            
            for part in part_history_global:
                if best is None or part.fitness < best.best_fitness:
                    best = part.clone()
                    print("Update gbest:", self.particle2dict(best))
            
            comm_inter.Barrier()

            if r_inter == 0:
                print("Best position so far:", best.position, "with args", best.fargs, "and fitness", best.best_fitness)
                # Write best parameter set to log.
                with open(log_path, "a+") as log: 
                    log.writelines("Best parameter set:"+" ".join(map("{:>15.4e}".format, best.position))+" with fitness"+"{:>15.4e}".format(best.best_fitness)+"\n")
                # Write parameter history to file.
                numpy.savetxt(params_path, fparams_history)
                # Checkpointing.
                with open(hist_path,"wb") as histp:
                    print("Dump particle history...")
                    pickle.dump(part_history_global, histp)
                    print("Done...")
                with open(best_path,"wb") as bestp:
                    print("Dump current gbest...")
                    pickle.dump(best, bestp)
                    print("Done...")
                print(fparams_history)

            # Propagate particle for next generation.
            self.updateParticle(PART, best, self.phi1, self.phi2)
            print("Particle updated for next generation...")
            print("Waiting for MPI barrier...")
            MPI.COMM_WORLD.Barrier()
            print("MPI barrier passed...")

        return dict([(k, v) for k, v in zip(self.bounds.keys(), best.position)]), None # Return best position for each hyperparameter.

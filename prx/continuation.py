# Copyright 2013 Ryan Volz

# This file is part of spyalg.

# Spyalg is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Spyalg is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with spyalg.  If not, see <http://www.gnu.org/licenses/>.

from itertools import izip

def dictzip(dkt):
    keys = dkt.keys()
    values = dkt.values()
    zipvalues = izip(*values)
    
    return (dict(zip(keys, v)) for v in zipvalues)

class Continuation(object):
    def __init__(self, func, **fixedargs):
        self.func = func
        self.fixedargs = fixedargs
    
    def __call__(self, **contargs):
        # each of the elements of the contargs dictionary should be an iterable
        kwargs = self.fixedargs.copy()
        
        def stateop(**state):
            for varargs in dictzip(contargs):
                kwargs.update(varargs)
                funcstateop = self.func(**kwargs)
                state = funcstateop(**state)
            
            return state
        
        return stateop
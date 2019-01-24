# VectorFields
Scripts to create vector fields that can be used in game engines like UnrealEngine 4.
Have a look at their [documentation](https://docs.unrealengine.com/en-us/Engine/Rendering/ParticleSystems/VectorFields).

## Usage
You have to know a bit of python and math.

The abstract base classes VectorField and VectorField2D can't be instantiated.

Instantiate a specific vector field, like ElectricDipole2D. You can use the plot-method to get a preview of the map.
You can then use the save_fga method to write the vector field to disk in FGA format.

### Special classes
#### ElectricDipole2D
ElectricDipole2D has 2 special methods to either normalize the vectors and lose all information on field strength or to clamp the field strength to a maximum value. This was necessary, because the physical properties of this field aren't visually pleasing.

#### CustomUV2D
This is a class for quick prototyping and generation of 2D vectorfields.

You provide custom functions to the constructor for creating the U and V vector components.  
These functions must take 2 parameters that will be substituted for the class' data members grid_x and grid_y.

__Examples:__

non-square wavy vector field 
```python
import numpy as np
from vectorfields import CustomUV2D
fu = lambda x,y: x/x  # =1. Flow in one direction.
fv = lambda x,y: np.sin(x)
vf = CustomUV2D(fu, fv, size=[8,2], resolution=[32,8])
vf.plot()
```
some diagonal flow thingy
```python
import numpy as np
from vectorfields import *   
fu = lambda x,y: np.cos(np.sqrt(np.abs(x)))  
fv = lambda x,y: np.cos(np.sqrt(np.abs(y)))  
vf = CustomUV2D(fu, fv)
vf.plot()
```  

## History
This little project started at a little game jam with the topic "electricity". I wanted to do something with particles and controlling their flow with formulas, but the available methods for creating vector fields at the time where either too complicated for this small task, or too time consuming or bound to purchasing a software license. Of course, this cannot compete with GUI software like VectorayGen, but it's free and open source.

## Development
Since it did what I needed it got stuck in early development stage. For example, there are no three-dimensional vector fields yet and there are no unit-tests.
I didn't yet take a look at Unity's vf format in order to write an export method.

I'd like to see people creating other vector fields with this and improve and advance the code.

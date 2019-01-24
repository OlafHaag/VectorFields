# VectorFields
Scripts to create vector fields that can be used in game engines like UnrealEngine 4.
Have a look at their [documentation](https://docs.unrealengine.com/en-us/Engine/Rendering/ParticleSystems/VectorFields).

## Usage
You have to know a bit of python and math.

The abstract base classes VectorField and VectorField2D can't be instantiated.

Instantiate a specific vector field, like ElectricDipole2D. You can use the plot-method to get a preview of the map.
You can then use the __save_fga method to write the vector field to disk in FGA format__.

### Special classes
#### ElectricDipole2D
ElectricDipole2D has 2 special methods to either normalize the vectors and lose all information on field strength or to clamp the field strength to a maximum value. This was necessary, because the physical properties of this field aren't visually pleasing.

#### CustomUV2D
This is a class for quick prototyping and generation of 2D vectorfields.

You provide custom functions to the constructor for creating the U and V vector components.  
These functions must take 2 parameters that will be substituted for the class' data members grid_x and grid_y.
The return value must be an array of the same shape as grid_x or grid_y respectively.

__Examples:__

non-square sine vector field 
```python
import numpy as np
from vectorfields import CustomUV2D
fu = lambda x,y: np.ones(x.shape)  # Flow in one direction.
fv = lambda x,y: np.sin(x)
vf = CustomUV2D(fu, fv, size=[8,2], resolution=[32,8])
vf.plot()
```
regular cosine whirls
```python
import numpy as np
from vectorfields import CustomUV2D
fu = lambda x,y: np.cos(y)
fv = lambda x,y: np.cos(x)
vf = CustomUV2D(fu, fv, size=16)
vf.plot()
```
"flowers"
```python
import numpy as np
from vectorfields import CustomUV2D
fu = lambda x,y: np.sin(y + x)
fv = lambda x,y: np.cos(x - y)
vf = CustomUV2D(fu, fv, size=12, resolution=48)
vf.plot()
```
some diagonal flow thingy
```python
import numpy as np
from vectorfields import CustomUV2D
fu = lambda x,y: np.cos(np.sqrt(np.abs(x)))  
fv = lambda x,y: np.cos(np.sqrt(np.abs(y)))  
vf = CustomUV2D(fu, fv)
vf.plot()
```  
[anvaka's](https://github.com/anvaka/fieldplay) square flow tiles (seriously, it's hard to find names for this stuff)
```python
import numpy as np
from vectorfields import CustomUV2D  
fu = lambda x,y: 2.0 * np.mod(np.floor(-y), 2.0) - 1.0
fv = lambda x,y: 2.0 * np.mod(np.floor(-x), -2.0) + 1.0
vf = CustomUV2D(fu, fv, size=5.9, resolution=24)
vf.plot()
```
beautiful twirls
```python
import numpy as np
from vectorfields import CustomUV2D 
fu = lambda x,y: np.cos(np.linalg.norm(np.vstack((x.flat, y.flat)), axis=0).reshape(x.shape))
fv = lambda x,y: np.cos(x-y)
vf = CustomUV2D(fu, fv, size=16)
vf.plot()
```
twirl column
```python
import numpy as np
from vectorfields import CustomUV2D 
fu = lambda x,y: np.sin(y)
fv = lambda x,y: x
vf = CustomUV2D(fu, fv, size=[12, 16], resolution=[24, 32])
vf.plot()
```
something a little bit more complex, "translated" from [anvaka's gallery](https://anvaka.github.io/fieldplay/?dt=0.02&fo=0.998&dp=0.009&cm=1&cx=0.21419999999999995&cy=-0.7710999999999997&w=55.970200000000006&h=55.970200000000006&code=v.x%20%3D%20min%28sin%28exp%28p.x%29%29%2Csin%28length%28p%29%29%29%3B%0Av.y%20%3D%20sin%28p.x%29%3B%0A%20%20) 
```python
import numpy as np
from vectorfields import CustomUV2D 
def fu(x,y):
    grid_norms = np.linalg.norm(np.vstack((x.flat, y.flat)), axis=0).reshape(x.shape)
    return np.minimum(np.sin(np.exp(x)),np.sin(grid_norms))
    
fv = lambda x,y: np.sin(x)
vf = CustomUV2D(fu, fv)
vf.plot()
```
How much time would it take to do that with effectors and forces?
```python
import numpy as np
from vectorfields import CustomUV2D 
fu = lambda x,y: np.cos(y**2)
fv = lambda x,y: np.cos(y*x)
vf = CustomUV2D(fu, fv, size=[24, 16], resolution=[48,32])
vf.plot()
```

## History
This little project started at a little game jam with the topic "electricity". I wanted to do something with particles and controlling their flow with formulas, but the available methods for creating vector fields at the time where either too complicated for this small task, or too time consuming or bound to purchasing a software license. Of course, this cannot compete with GUI software like VectorayGen, but it's free and open source.

## Development
Since it did what I needed it got stuck in early development stage. For example, there are no three-dimensional vector fields yet and there are no unit-tests.
I didn't yet take a look at Unity's vf format in order to write an export method.

I'd like to see people creating other vector fields with this and improve and advance the code.

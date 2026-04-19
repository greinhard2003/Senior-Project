import random
from cube import Cube, moves
from encode import encode_cube

def random_scramble(cube: Cube, depth: int) -> Cube:
    for _ in range(depth):
        cube = cube.apply_move(random.choice(moves))
    return cube

def generate_sample(max_depth: int):
    depth = random.randint(1, max_depth)
    cube = random_scramble(Cube(), depth)

    x = encode_cube(cube)
    y = depth
    return x, y
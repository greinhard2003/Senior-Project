from dataclasses import dataclass, field
import random

@dataclass(frozen=True)
class Move:
    cp: tuple
    co: tuple
    ep: tuple
    eo: tuple

@dataclass()
class Cube:
    #Cube state modeled using the cubie model
    """
    Corner Pieces:
    0: URF 1: UFL 2: ULB 3: UBR
    4: DFR 5: DLF 6: DBL 7: DRB

    Edge Pieces:
    0: UR   1: UF    2: UL   3: UB
    4: DR   5: DF    6: DL   7: DB
    8: FR   9: FL   10: BL  11: BR

    """
    #Corner Positions
    cp: list = field(default_factory=lambda: list(range(8)))
    #Corner Orientations
    co: list = field(default_factory=lambda: [0]*8)
    #Edge Positions
    ep: list = field(default_factory=lambda: list(range(12)))
    #Edge Orientations
    eo: list = field(default_factory=lambda: [0]*12)

    def apply_move(self, move: Move):
        return Cube(
            cp=[self.cp[move.cp[i]] for i in range(8)],
            co=[(self.co[move.cp[i]] + move.co[i]) % 3 for i in range(8)],
            ep=[self.ep[move.ep[i]] for i in range(12)],
            eo=[(self.eo[move.ep[i]] + move.eo[i]) % 2 for i in range(12)]
        )

    def is_solved(self) -> bool:
        return (self.cp == list(range(8)) and
                all(v == 0 for v in self.co) and
                self.ep == list(range(12)) and
                all(v == 0 for v in self.eo))
    def __repr__(self):
        res = "--------------Cube state:--------------\n"
        res += f"Corner Positions: {self.cp}\n"
        res += f"Corner Orientation: {self.co}\n"
        res += f"Edge Positions: {self.ep}\n"
        res += f"Edge Orientation: {self.eo}\n"
        return res

def compose(a: Move, b: Move) -> Move:
    #Using this to extrapolate CCW moves once at runtime then it will
    #be constant time afterward
    cp = tuple(b.cp[a.cp[i]] for i in range(8))
    co = tuple((b.co[a.cp[i]] + a.co[i]) % 3 for i in range(8))
    ep = tuple(b.ep[a.ep[i]] for i in range(12))
    eo = tuple((b.eo[a.ep[i]] + a.eo[i]) % 2 for i in range(12))
    return Move(cp, co, ep, eo)

R = Move(
    cp = (4, 1, 2, 0, 7, 5, 6, 3),
    co = (2, 0, 0, 1, 1, 0, 0, 2),
    ep = (8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0),
    eo = (0,)*12
)
R_PRIME = compose(R, compose(R, R))

L = Move(
    cp = (0, 2, 6, 3, 4, 1, 5, 7),
    co=(0, 1, 2, 0, 0, 2, 1, 0),
    ep = (0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11),
    eo = (0,)*12
)

L_PRIME = compose(L, compose(L, L))



F = Move(
    cp = (1, 5, 2, 3, 0, 4, 6, 7),
    co = (2, 1, 0, 0, 1, 2, 0, 0),
    ep = (0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11),
    eo = (0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0)
)

F_PRIME = compose(F, compose(F, F))

B = Move(
    cp=(0, 1, 3, 7, 4, 5, 2, 6),
    co=(0, 0, 2, 1, 0, 0, 1, 2),
    ep=(0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7),
    eo=(0, 0, 0, 1 , 0, 0, 0, 1, 0, 0, 1, 1)
)

B_PRIME = compose(B, compose(B, B))

U = Move(
    cp=(3, 0, 1, 2, 4, 5, 6, 7),
    co=(0, 0, 0, 0, 0, 0, 0, 0),
    ep=(3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11),
    eo=(0,)*12
)

U_PRIME = compose(U, compose(U, U))

D = Move(
    cp=(0, 1, 2, 3, 5, 6, 7, 4),
    co=(0, 0, 0, 0, 0, 0, 0, 0),
    ep=(0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11),
    eo=(0,)*12
)

D_PRIME = compose(D, compose(D, D))

"""
MOVE TABLE:
0: R     1: R'      2: L     3: L'
4: F     5: F'      6: B     7: B'
8: U     9: U'     10: D    11: D'
"""
move_names = ['R', 'R\'', 'L', 'L\'', 'F', 'F\'', 'B', 'B\'', 'U', 'U\'', 'D', 'D\'']
moves = [R, R_PRIME, L, L_PRIME, F, F_PRIME, B, B_PRIME, U, U_PRIME, D, D_PRIME]



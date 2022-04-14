import inspect
from typing import Tuple, List, Dict, Union

import numpy as np

from library.common.utils import Objectless

# discretization points for theta and phi
n = 100
fill = True


class Shape(Objectless):

    @classmethod
    def inner_classes_list(cls):
        return [d for d in dir(Shape) if inspect.isclass(getattr(Shape, d)) and d != '__class__']

    @classmethod
    def make_shapes(cls, **kwargs) -> List[Tuple[int, int, int]]:
        ret = list()
        for sh in kwargs.keys():
            inner_classes = cls.inner_classes_list()
            sh_name = sh.split(' ')[0].title()
            if not sh_name in inner_classes:
                raise Exception(f"Shape {sh_name} unknown. Available shapes: {inner_classes}")
            kwargs[sh] = {k.lower(): v for k, v in kwargs[sh].items()}
            ret += eval(f"Shape.{sh_name.title()}(**kwargs[sh])")
        return ret

    class Cuboid:
        def __new__(cls, **kwargs) -> List[Tuple[int, int, int]]:
            # x_width: int, y_depth: int, z_height: int, origin: Tuple[int, int, int] = (0, 0, 0), fill: bool = True
            params = ["width", "depth", "height", "origin"]
            # check if kwargs contains all elements in params
            assert all(elem in kwargs.keys() for elem in params), f"Missing parameters {params}."
            width = kwargs["width"]
            depth = kwargs["depth"]
            height = kwargs["height"]
            assert width > 0 and depth > 0 and height > 0, "Cuboid width, depth and height must be positive integers."
            xo, yo, zo = tuple(kwargs["origin"])

            xvalues = range(xo, xo + width)
            yvalues = range(yo, yo + depth)
            zvalues = range(zo, zo + height)
            x, y, z = np.meshgrid(xvalues, yvalues, zvalues, indexing='ij')
            x = [int(_x) for _x in x.ravel()]
            y = [int(_y) for _y in y.ravel()]
            z = [int(_z) for _z in z.ravel()]
            return list(set(zip(x, y, z)))

    class Cylinder:
        def __new__(cls, **kwargs):
            params = ["center", "radius", "height"]
            # check if kwargs contains all elements in params
            assert all(elem in kwargs.keys() for elem in params), f"Missing parameters {params}."
            center = tuple(kwargs["center"])
            xc, yc, zc = center
            radius = kwargs["radius"]
            assert radius > 0, "Radius must be a positive integer."
            height = kwargs["height"]
            assert height > 0, "Height must be a positive integer."
            theta = np.linspace(0, 2 * np.pi, n)
            z = [int(_z) for _z in np.linspace(zc, zc + height, height)]
            x = [int(_x) for _x in np.rint([xc + rho * np.cos(theta) for rho in range(radius)]).astype(int).ravel()]
            y = [int(_y) for _y in np.rint([yc + rho * np.sin(theta) for rho in range(radius)]).astype(int).ravel()]
            coords = list()
            for coord in list(set(zip(x, y))):
                coords += [(*coord, _z) for _z in z]
            return list(set(coords))

    class Sphere:
        def __new__(cls, **kwargs):
            params = ["center", "radius"]
            # check if kwargs contains all elements in params
            assert all(elem in kwargs.keys() for elem in params), f"Missing parameters {params}."
            center = tuple(kwargs["center"])
            radius = kwargs["radius"]
            assert radius > 0, "Radius must be a positive integer."
            theta = np.linspace(0, 2 * np.pi, n)
            phi = np.linspace(0, np.pi, n)
            xc, yc, zc = center
            if fill:
                x = [int(_x) for _x in
                     np.rint([xc + rho * np.outer(np.cos(theta), np.sin(phi))
                              for rho in range(radius)]).astype(int).ravel()]
                y = [int(_y) for _y in
                     np.rint([yc + rho * np.outer(np.sin(theta), np.sin(phi))
                              for rho in range(radius)]).astype(int).ravel()]
                z = [int(_z) for _z in
                     np.rint([zc + rho * np.outer(np.ones(np.size(theta)), np.cos(phi))
                              for rho in range(radius)]).astype(int).ravel()]
            else:
                x = [int(_x) for _x in xc + radius * np.outer(np.cos(theta), np.sin(phi))]
                y = [int(_y) for _y in yc + radius * np.outer(np.sin(theta), np.sin(phi))]
                z = [int(_z) for _z in zc + radius * np.outer(np.ones(np.size(theta)), np.cos(phi))]
            return list(set(zip(x, y, z)))

    class Torus:
        def __new__(cls, **kwargs):
            # center: Tuple[int, int, int] = (0, 0, 0), major_radius: int = 3, minor_radius: int = 2, n=100
            params = ["center", "major_radius", "minor_radius"]
            # check if kwargs contains all elements in params
            assert all(elem in kwargs.keys() for elem in params), f"Missing parameters {params}."
            center = tuple(kwargs["center"])
            major_radius = kwargs["major_radius"]
            minor_radius = kwargs["minor_radius"]
            assert major_radius > 0 and minor_radius > 0, "Major radius and minor radius must be positive integers."
            theta = np.linspace(0, 2 * np.pi, n)
            phi = np.linspace(0, 2 * np.pi, n)
            theta, phi = np.meshgrid(theta, phi)
            xc, yc, zc = center
            x = [int(_x) for _x in np.rint(xc + (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)).astype(int).ravel()]
            y = [int(_y) for _y in np.rint(yc + (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)).astype(int).ravel()]
            z = [int(_z) for _z in np.rint(zc + minor_radius * np.sin(theta)).astype(int).ravel()]

            return list(set(zip(x, y, z)))

    class Cardioid:
        def __new__(cls, **kwargs):
            params = ["center", "height", "alpha"]
            # check if kwargs contains all elements in params
            assert all(elem in kwargs.keys() for elem in params), f"Missing parameters {params}."
            center = tuple(kwargs["center"])
            height = int(kwargs["height"])
            alpha = kwargs["alpha"]
            assert height > 0, "Height must be greater than 0."
            assert alpha > 0, "Alpha must be greater than 0."
            theta = np.linspace(0, 2 * np.pi, n)
            phi = np.linspace(0, np.pi, n)
            theta, phi = np.meshgrid(theta, phi)
            xc, yc, zc = center
            #np.outer(np.cos(theta), np.sin(phi)
            # era 16 dove adesso è 8
            x = [int(_x)
                 for _x in xc + alpha * 2.5 * (16 * np.power(np.sin(theta) * np.sin(phi), 3)).ravel()]
            # era 13 dove adesso è 3
            y = [int(_y)
                 for _y in yc + alpha * 0.5 * (15 * (np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)) * -np.sin(phi)).ravel()]
            z = [center[2] for _x in range(len(x)) for _ in range(height)]
            return list(set(zip(x, y, z)))


def get_mask(template: Union[Dict, Tuple[int, int, int]], mask: Union[Dict, List[Tuple[int, int, int]]]) -> Dict[Tuple[int, int, int], int]:
    assert isinstance(template, tuple), NotImplementedError("'template' parameter must be the tuple of integers (x_width, y_depth, z_height).")
    assert all([type(_i) == int for _i in template])
    if isinstance(mask, dict):
        ls = Shape.make_shapes(**mask)
        ls = crop_coords(ls, template)
        d2 = {_k: 1 for _k in ls}
    else:
        d2 = {_k: 1 for _k in mask}
    x, y, z = template
    d1: Dict[Tuple[int, int, int], int] = dict()
    [d1.update({(i, j, k): 0}) for i in range(x) for j in range(y) for k in range(z)]
    return d1 | d2


def crop_coords(coords: List[Tuple[int, int, int]], margins: Tuple[int, int, int]):
    return [_t for _t in coords if all((a < b) for a, b in zip(_t, margins))]


def unzip(coord: List[Tuple[int, int, int]]):
    return map(list, zip(*coord))

def draw(coord: Dict[Tuple[int, int, int], int], fig_name="shape.png", margins: Tuple[int, int, int]=None):
    import seaborn as sns
    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4, "ztick.major.size": 4})
    from matplotlib import pyplot as plt
    import pandas as pd

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(90, 90)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(0, 0)

    K = list(coord.keys())
    V = list(coord.values())
    (X, Y, Z) = zip(*K)
    data = pd.DataFrame({"X Value": X, "Y Value": Y, "Z Value": Z, "Category": V})
    groups = data.groupby("Category")
    for name, group in groups:
        if name == 1:
            ax1.scatter(group["X Value"], group["Y Value"], group["Z Value"], marker=".", label='target')
            ax2.scatter(group["X Value"], group["Y Value"], group["Z Value"], marker=".", label='target')
    #ax1.scatter(data["X Value"], data["Y Value"], data["Z Value"], marker='o', label="diocane")
    #ax2.scatter(data["X Value"], data["Y Value"], data["Z Value"], marker='o', label="diocane")
    ax1.scatter(0, 0, 0, marker='o', color='red', label='(0,0,0)')
    ax2.scatter(0, 0, 0, marker='o', color='red', label='(0,0,0)')
    ax1.scatter(margins[0], margins[1], 0, marker='o', color='green', label=f'({margins[0]},{margins[1]},0)')
    ax2.scatter(margins[0], margins[1], 0, marker='o', color='green', label=f'({margins[0]},{margins[1]},0)')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    if margins:
        #plt.xlim(0, margins[0]+1)
        #plt.ylim(0, margins[1]+1)
        ax1.set_xlim(0, margins[0] + 1)
        ax2.set_xlim(0, margins[0] + 1)
        ax1.set_ylim(0, margins[1] + 1)
        ax2.set_ylim(0, margins[1] + 1)
        ax1.set_zlim(0, margins[2] + 1)
        ax2.set_zlim(0, margins[2] + 1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    plt.savefig(fig_name)
    plt.close()


def main():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    # plt.gca().set_aspect('equal', adjustable='box')
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-100, 100)
    ax1.set_zlim(-100, 100)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(0, 50)
    ax2.set_zlim(0, 50)

    # sphere 1
    sphere1 = Shape.Sphere(center=(0, 0, 0), radius=25)
    X1, Y1, Z1 = unzip(sphere1)
    ax1.scatter(X1, Y1, Z1, marker='.', color='b', alpha=0.1)

    # sphere 2
    sphere2 = Shape.Sphere(center=(10, 30, 20), radius=5)
    X2, Y2, Z2 = unzip(sphere2)
    ax1.scatter(X2, Y2, Z2, marker='.', color='r', alpha=0.1)

    # torus
    torus1 = Shape.Torus(center=(50, 50, 50), major_radius=15, minor_radius=5)
    X3, Y3, Z3 = unzip(torus1)
    ax1.scatter(X3, Y3, Z3, marker='.', color='g', alpha=0.1)

    # cuboid
    cuboid1 = Shape.Cuboid(width=30, depth=30, height=10, origin=(-15, 10, 0))
    X4, Y4, Z4, = unzip(cuboid1)
    ax2.scatter(X4, Y4, Z4, marker='.', color='orange', alpha=0.1)
    torus2 = Shape.Torus(center=(0, 30, 20), major_radius=15, minor_radius=5)
    X5, Y5, Z5 = unzip(torus2)
    ax2.scatter(X5, Y5, Z5, marker='.', color='grey', alpha=0.1)

    cylinder1 = Shape.Cylinder(center=(0, 30, 30), radius=15, height=50)
    X6, Y6, Z6 = unzip(cylinder1)
    ax1.scatter(X6, Y6, Z6, marker='.', color='royalblue', alpha=0.1)

    plt.savefig("geometry.png")
    plt.close()


def shape_tst():
    import matplotlib.pyplot as plt

    kwargs = {"CUBOID": {"width": 40, "DEPTH": 40, "HEIGHT": 15, "ORIGIN": (0, 0, 0)},
              "SPHERE": {"CENTER": (60, 60, 60), "RADIUS": 20},
              "cylinder": {"center": (30, -75, -30), "radius": 15, "height": 20},
              "torus": {"center": (-70, -70, 50), "major_radius": 30, "minor_radius": 10}}

    for _name, _descr in kwargs.items():
        res = Shape.make_shapes(**kwargs)
        x, y, z = unzip(res)
        fig = plt.figure()
        # plt.gca().set_aspect('equal', adjustable='box')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)
        ax.scatter(x, y, z, marker='.', color='grey', alpha=0.1)
        plt.savefig(_name + ".png")
        plt.close()


def target_tst():
    margins = (200, 200, 200)
    targets = {
        "STRIPE": {"CUBOID": {"width": 200, "DEPTH": 80, "HEIGHT": 20, "ORIGIN": (0, 60, 3)}},
        "CIRCLE": {"CYLINDER": {"center": (100, 100, 3), "radius": 60, "height": 1}}
    }
    for name, descr in targets.items():
        _mask = get_mask(margins, descr)
        draw(_mask, fig_name=f"{name}_target.png", margins=margins)


if __name__ == "__main__":
    target_tst()

import numpy as np
from scipy.spatial import Delaunay


class GRID:
    def __init__(self, pts=None, face=None):
        """
        :param pts: Nx2
        :param face: Fx3
        """
        if pts is None:
            pass
        else:
            self.v = pts
            if face is not None:
                self.adj = self.buildAdj(face)
            else:
                # delaunay triangulation
                tri = Delaunay(self.v)
                face = tri.simplices
                self.buildAdj(face)

    def buildAdj(self, face):
        """
        face: Fx3
        """
        self.adj = np.zeros((len(self.v), len(self.v)), dtype=np.int64)
        self.adj[face[:, 0], face[:, 1]] = 1
        self.adj[face[:, 0], face[:, 2]] = 1
        self.adj[face[:, 1], face[:, 2]] = 1
        self.adj = np.maximum(self.adj, self.adj.T)

    def load(self, filename):
        with open(filename, "r") as fin:
            lines = fin.readlines()
        self.v = []
        self.e = []
        for li in lines:
            if li.split()[0]=='v':
                self.v.append(np.array([int(li.split()[1]), int(li.split()[2])]))
            elif li.split()[0]=="e":
                self.e.append(np.array([int(li.split()[1]), int(li.split()[2])]))
        self.v = np.array(self.v)
        self.e = np.array(self.e)
        self.adj = np.zeros((len(self.v), len(self.v)), dtype=np.int64)
        self.adj[self.e[:,0], self.e[:,1]] = 1
        self.adj = np.maximum(self.adj, self.adj.T)


    def save(self, filename):
        self.e = np.argwhere(np.triu(self.adj))
        with open(filename, "w") as fout:
            for v in self.v:
                fout.write(f"v {v[0]} {v[1]}\n")
            for e in self.e:
                fout.write(f"e {e[0]} {e[1]}\n")

